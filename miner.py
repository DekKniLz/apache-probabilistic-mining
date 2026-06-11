import os
import time
import json
import csv
import subprocess
import shutil
from tree_sitter import Language, Parser
import tree_sitter_java

MAX_FILE_SIZE_MB = 2  
TARGET_STRUCTURES = {"BloomFilter", "IFilter", "HyperLogLog", "CountMinSketch", "CuckooFilter", "HyperLogLogPlus"}

parser = Parser()
parser.language = Language(tree_sitter_java.language())

def walk_ast(node, filepath, repo_path, repo_name, findings):
    if node.type == 'import_declaration':
        text = node.text.decode('utf8')
        if any(struct in text for struct in TARGET_STRUCTURES):
            findings.append({
                "repository": repo_name, "file_name": os.path.basename(filepath),
                "context": "IMPORT", "target_structure": text.strip().replace(';', ''),
                "line_number": node.start_point[0] + 1, "relative_path": os.path.relpath(filepath, repo_path)
            })
    elif node.type == 'object_creation_expression':
        type_node = node.child_by_field_name('type')
        if type_node and type_node.text.decode('utf8') in TARGET_STRUCTURES:
            findings.append({
                "repository": repo_name, "file_name": os.path.basename(filepath),
                "context": "INSTANTIATION", "target_structure": type_node.text.decode('utf8'),
                "line_number": type_node.start_point[0] + 1, "relative_path": os.path.relpath(filepath, repo_path)
            })
    for child in node.children: walk_ast(child, filepath, repo_path, repo_name, findings)

def mine_repository(repo_path):
    java_files = [os.path.join(r, f) for r, d, files in os.walk(repo_path) for f in files if f.endswith('.java') and os.path.getsize(os.path.join(r, f)) < (MAX_FILE_SIZE_MB * 1024 * 1024)]
    repo_name = os.path.basename(os.path.normpath(repo_path))
    findings = []
    for fp in java_files:
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                tree = parser.parse(bytes(f.read(), "utf8"))
                walk_ast(tree.root_node, fp, repo_path, repo_name, findings)
        except Exception: pass
    return findings

def clone_and_mine():
    with open("repos.txt", 'r') as f: urls = [l.strip() for l in f if l.strip()]
    all_findings, workspace = [], "./repos_workspace"
    os.makedirs(workspace, exist_ok=True)

    for i, url in enumerate(urls):
        repo_name = url.split('/')[-1].replace('.git', '')
        target = os.path.join(workspace, repo_name)
        print(f"--- Processing [{i+1}/{len(urls)}]: {repo_name} ---")
        try:
            subprocess.run(["git", "clone", "--depth", "1", url, target], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            all_findings.extend(mine_repository(target))
        except Exception: pass
        finally:
            if os.path.exists(target): shutil.rmtree(target, ignore_errors=True)
            
    os.makedirs("./output", exist_ok=True)
    if all_findings:
        with open("./output/dataset.csv", 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=all_findings[0].keys())
            w.writeheader(); w.writerows(all_findings)
    print("\n[SUCCESS] Mining complete. Dataset saved to ./output/dataset.csv")

if __name__ == "__main__": clone_and_mine()