import os
import time
import json
import csv
import subprocess
import shutil
from tree_sitter import Language, Parser
import tree_sitter_java

# --- CONFIGURATION & SECURITY THRESHOLDS ---
BATCH_SIZE = 100
THROTTLE_DELAY_SEC = 0.5
MAX_FILE_SIZE_MB = 2  

# Initialize the Java language parser
JAVA_LANGUAGE = Language(tree_sitter_java.language())
parser = Parser()
parser.language = JAVA_LANGUAGE

# The specific probabilistic data structures we are hunting for
TARGET_STRUCTURES = {"BloomFilter", "IFilter", "HyperLogLog", "CountMinSketch", "CuckooFilter"}


def is_safe_path(base_path, target_path):
    """
    Prevents Path Traversal vulnerabilities by verifying that the target
    file resides strictly inside the allocated repository directory.
    """
    base_abs = os.path.abspath(base_path)
    target_abs = os.path.abspath(target_path)
    return target_abs.startswith(base_abs)


def export_results(findings, output_dir="./output"):
    """
    Exports the gathered metadata into structured JSON and CSV datasets.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(findings, f, indent=4)
        
    csv_path = os.path.join(output_dir, "dataset.csv")
    if findings:
        keys = findings[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(findings)
            
    print(f"\n[INFO] Global dataset exported successfully to: {output_dir}/")


def walk_ast(node, filepath, repo_path, repo_display_name, findings):
    """
    Recursively walks the Abstract Syntax Tree (AST) bypassing the unstable 
    Query API entirely. It evaluates node types directly for robust extraction.
    """
    # 1. Detect Imports
    if node.type == 'import_declaration':
        captured_text = node.text.decode('utf8')
        # Check if any target structure is mentioned in the import statement
        if any(struct in captured_text for struct in TARGET_STRUCTURES):
            clean_text = captured_text.strip().replace(';', '')
            record = {
                "repository": repo_display_name,
                "file_name": os.path.basename(filepath),
                "context": "IMPORT",
                "target_structure": clean_text,
                "line_number": node.start_point[0] + 1,
                "relative_path": os.path.relpath(filepath, repo_path)
            }
            findings.append(record)
            print(f"[*] IMPORT detected: {clean_text} in {record['file_name']} (Line: {record['line_number']})")
            
    # 2. Detect Object Instantiations (e.g., new BloomFilter())
    elif node.type == 'object_creation_expression':
        type_node = node.child_by_field_name('type')
        if type_node:
            type_text = type_node.text.decode('utf8')
            if type_text in TARGET_STRUCTURES:
                record = {
                    "repository": repo_display_name,
                    "file_name": os.path.basename(filepath),
                    "context": "INSTANTIATION",
                    "target_structure": type_text,
                    "line_number": type_node.start_point[0] + 1,
                    "relative_path": os.path.relpath(filepath, repo_path)
                }
                findings.append(record)
                print(f"[*] INSTANTIATION detected: {type_text} in {record['file_name']} (Line: {record['line_number']})")
    
    # 3. Recurse down the tree
    for child in node.children:
        walk_ast(child, filepath, repo_path, repo_display_name, findings)


def mine_repository(repo_path):
    java_files = []
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.java'):
                full_path = os.path.join(root, file)
                
                if not is_safe_path(repo_path, full_path):
                    continue
                    
                if os.path.getsize(full_path) > (MAX_FILE_SIZE_MB * 1024 * 1024):
                    continue
                    
                java_files.append(full_path)

    total_files = len(java_files)
    repo_display_name = os.path.basename(os.path.normpath(repo_path))
    print(f"[INFO] Launching AST parsing engine on {total_files} Java files from '{repo_display_name}'...")

    findings = []

    for i, filepath in enumerate(java_files):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = bytes(f.read(), "utf8")
                
            tree = parser.parse(source_code)
            
            # Use the bulletproof recursive AST walker instead of the Query API
            walk_ast(tree.root_node, filepath, repo_path, repo_display_name, findings)
                
        except Exception as e:
            print(f"[ERROR] Engine failed to parse {filepath}: {e}")

        # CPU Throttling
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(THROTTLE_DELAY_SEC)
            
    print(f"[SUCCESS] Completed mining for '{repo_display_name}'. Found {len(findings)} references.")
    return findings


def clone_and_mine(repo_urls_file):
    if not os.path.exists(repo_urls_file):
        print(f"[ERROR] Configuration target file '{repo_urls_file}' was not found.")
        return

    with open(repo_urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    total_repos = len(urls)
    print(f"\n[INFO] Automation pipeline initialized. {total_repos} repositories in queue.\n")

    all_findings = []
    workspace_dir = "./repos_workspace"
    
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)

    for index, url in enumerate(urls):
        repo_name = url.split('/')[-1].replace('.git', '')
        target_path = os.path.join(workspace_dir, repo_name)
        
        print(f"\n--- Processing Project [{index + 1}/{total_repos}]: {repo_name} ---")
        
        print(f"[INFO] Cloning remote repository: {url}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, target_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Network execution failed for {repo_name}. Skipping repository.")
            continue

        repo_findings = mine_repository(target_path)
        if repo_findings:
            all_findings.extend(repo_findings)

        print(f"[INFO] Wiping workspace directory for: {repo_name}")
        try:
            shutil.rmtree(target_path)
        except Exception as e:
            print(f"[ERROR] Resource lock detected. Could not remove directory {target_path}: {e}")

    if os.path.exists(workspace_dir):
        try:
            shutil.rmtree(workspace_dir)
        except Exception:
            pass

    print("\n[INFO] Pipeline routine finished. Compiling global analytics...")
    export_results(all_findings)



if __name__ == "__main__":
    target_config_file = "repos.txt"
    clone_and_mine(target_config_file)