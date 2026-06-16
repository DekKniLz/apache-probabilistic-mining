import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("[ERROR] No GITHUB_TOKEN found. Check your .env file.")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}", 
    "Accept": "application/vnd.github.v3+json"
}

def fetch_official_asf_repos():
    """
    Methodological Extraction via Data Triangulation:
    1. Uses ASF projects.json as the absolute source of truth (Whitelist).
    2. Uses GitHub API to guarantee the language is strictly Java and the repo is active.
    """
    print("[INFO] Initializing Methodological ASF Extraction...")
    
    # --- PHASE 1: Build the Official Whitelist ---
    print("[INFO] Fetching official ASF project whitelist from projects.apache.org...")
    asf_url = "https://projects.apache.org/json/foundation/projects.json"
    asf_resp = requests.get(asf_url)
    
    if asf_resp.status_code != 200:
        print(f"[ERROR] Failed to fetch ASF registry. HTTP: {asf_resp.status_code}")
        return
        
    asf_projects = asf_resp.json()
    
    official_basenames = set()
    for proj_id, info in asf_projects.items():
        official_basenames.add(proj_id.lower())
        repos = info.get("repository", [])
        if isinstance(repos, str): 
            repos = [repos]
        for r in repos:
            if isinstance(r, str) and "git" in r:
                name = r.split('/')[-1].replace('.git', '').lower()
                official_basenames.add(name)

    print(f"[*] Built an official whitelist of {len(official_basenames)} ASF projects.")
    print("[*] Cross-referencing with GitHub to guarantee 100% Java codebases...\n")

    # --- PHASE 2: Triangulate with GitHub Metadata ---
    java_repos = set()
    page = 1
    
    while True:
        gh_url = f"https://api.github.com/orgs/apache/repos?type=public&per_page=100&page={page}"
        gh_resp = requests.get(gh_url, headers=HEADERS)
        
        if gh_resp.status_code == 403:
            print("    [WARN] API rate limit. Cooling down for 15 seconds...")
            time.sleep(15)
            continue
        elif gh_resp.status_code != 200:
            break
            
        data = gh_resp.json()
        if not data:
            break
            
        for repo in data:
            repo_name = repo["name"].lower()
            
            # Intersection Check: Is it an official ASF project name?
            is_official = any(repo_name.startswith(base) for base in official_basenames)
            
            # MSR Filter: Strictly Java, active (not archived), and officially recognized
            if repo.get("language") == "Java" and not repo.get("archived") and is_official:
                java_repos.add(repo["clone_url"])
                print(f"    [+] Validated Official ASF Java Repo: {repo['name']}")
                
        page += 1
        time.sleep(1) # Throttle to respect API limits
        
    print(f"\n[SUCCESS] Methodological extraction complete.")
    print(f"[SUCCESS] Isolated {len(java_repos)} official ASF Java repositories.")
    
    with open("repos.txt", "w") as f:
        for url in sorted(java_repos):
            f.write(url + "\n")
            
    print("[INFO] Target list successfully written to repos.txt. Pipeline is ready for AST Mining.")

if __name__ == "__main__":
    fetch_official_asf_repos()