import csv
import json
import os
import shutil
import subprocess
from tree_sitter import Language, Parser
import tree_sitter_java

MAX_FILE_SIZE_MB = 2
CONFIG_PATH = "./pds_targets.json"


def load_target_structures(config_path=CONFIG_PATH):
    """Loads configurable target structure names from a JSON file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            items = payload.get("target_structures", [])
            if items:
                return {str(item) for item in items}
        except Exception as exc:
            print(f"[WARN] Unable to load {config_path}: {exc}")

    return {
        "BloomFilter",
        "IFilter",
        "HyperLogLog",
        "CountMinSketch",
        "CuckooFilter",
        "HyperLogLogPlus",
    }


TARGET_STRUCTURES = load_target_structures()

parser = Parser()
parser.language = Language(tree_sitter_java.language())


def _matches_target(text):
    normalized = str(text).lower()
    return any(struct.lower() in normalized for struct in TARGET_STRUCTURES)


def walk_ast(node, filepath, repo_path, repo_name, findings, file_imports):
    if node.type == "import_declaration":
        text = node.text.decode("utf8").strip()
        if _matches_target(text):
            file_imports.append(text.strip().replace(";", ""))
            findings.append(
                {
                    "repository": repo_name,
                    "file_name": os.path.basename(filepath),
                    "context": "IMPORT",
                    "target_structure": text.strip().replace(";", ""),
                    "line_number": node.start_point[0] + 1,
                    "relative_path": os.path.relpath(filepath, repo_path),
                    "import_hint": "",
                }
            )
    elif node.type == "object_creation_expression":
        type_node = node.child_by_field_name("type")
        if type_node:
            type_name = type_node.text.decode("utf8")
            if type_name in TARGET_STRUCTURES or _matches_target(type_name):
                import_hint = next(
                    (item for item in reversed(file_imports) if type_name.lower() in item.lower()),
                    "",
                )
                findings.append(
                    {
                        "repository": repo_name,
                        "file_name": os.path.basename(filepath),
                        "context": "INSTANTIATION",
                        "target_structure": type_name,
                        "line_number": type_node.start_point[0] + 1,
                        "relative_path": os.path.relpath(filepath, repo_path),
                        "import_hint": import_hint,
                    }
                )
    for child in node.children:
        walk_ast(child, filepath, repo_path, repo_name, findings, file_imports)


def mine_repository(repo_path):
    java_files = [
        os.path.join(r, f)
        for r, d, files in os.walk(repo_path)
        for f in files
        if f.endswith(".java") and os.path.getsize(os.path.join(r, f)) < (MAX_FILE_SIZE_MB * 1024 * 1024)
    ]
    repo_name = os.path.basename(os.path.normpath(repo_path))
    findings = []
    parse_failures = []

    for fp in java_files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as handle:
                tree = parser.parse(bytes(handle.read(), "utf8"))
                walk_ast(tree.root_node, fp, repo_path, repo_name, findings, [])
        except Exception as exc:
            parse_failures.append({"repository": repo_name, "file": os.path.basename(fp), "error": str(exc)})

    return findings, parse_failures


def clone_and_mine():
    with open("repos.txt", "r", encoding="utf-8") as handle:
        urls = [line.strip() for line in handle if line.strip()]

    all_findings, all_parse_failures = [], []
    workspace = "./repos_workspace"
    os.makedirs(workspace, exist_ok=True)

    for i, url in enumerate(urls):
        repo_name = url.split("/")[-1].replace(".git", "")
        target = os.path.join(workspace, repo_name)
        print(f"--- Processing [{i + 1}/{len(urls)}]: {repo_name} ---")
        try:
            subprocess.run(["git", "clone", "--depth", "1", url, target], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            findings, failures = mine_repository(target)
            all_findings.extend(findings)
            all_parse_failures.extend(failures)
        except Exception as exc:
            print(f"[WARN] Failed to process {repo_name}: {exc}")
        finally:
            if os.path.exists(target):
                shutil.rmtree(target, ignore_errors=True)

    os.makedirs("./output", exist_ok=True)
    if all_findings:
        with open("./output/dataset.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_findings[0].keys())
            writer.writeheader()
            writer.writerows(all_findings)

    with open("./output/parse_failures.json", "w", encoding="utf-8") as handle:
        json.dump(all_parse_failures, handle, indent=2, ensure_ascii=False)

    print("\n[SUCCESS] Mining complete. Dataset saved to ./output/dataset.csv")
    print("[SUCCESS] Parse failures report saved to ./output/parse_failures.json")


if __name__ == "__main__":
    clone_and_mine()