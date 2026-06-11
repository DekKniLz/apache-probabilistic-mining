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

SEARCH_TERMS = ["BloomFilter", "HyperLogLog", "CountMinSketch", "CuckooFilter", "HyperLogLogPlus"]
MIN_STARS = 100
MIN_FORKS = 25

def get_repo_details(full_name):
    url = f"https://api.github.com/repos/{full_name}"
    resp = requests.get(url, headers=HEADERS)
    return resp.json() if resp.status_code == 200 else None

def fetch_rigorous_wild_repos():
    print("[INFO] Initializing 'Rigorous In-the-Wild' MSR Radar...")
    
    rigorous_repos = set()
    evaluated_repos = set()
    
    for term in SEARCH_TERMS:
        print(f"\n[*] Hunting for '{term}' in real-world Java code...")
        query = f'"{term}" in:file language:java'
        page = 1
        
        while page <= 3:  
            url = f"https://api.github.com/search/code?q={query}&per_page=100&page={page}"
            response = requests.get(url, headers=HEADERS)
            
            if response.status_code == 403:
                time.sleep(15)
                continue
            elif response.status_code != 200:
                break
                
            items = response.json().get("items", [])
            if not items: break
                
            for item in items:
                full_name = item["repository"]["full_name"]
                if full_name in evaluated_repos: continue
                evaluated_repos.add(full_name)
                
                repo_data = get_repo_details(full_name)
                if repo_data:
                    stars = repo_data.get("stargazers_count", 0)
                    forks = repo_data.get("forks_count", 0)
                    if stars >= MIN_STARS and forks >= MIN_FORKS and not repo_data.get("fork", False):
                        rigorous_repos.add(repo_data["clone_url"])
                        print(f"    [+] Passed: {full_name} (Stars: {stars} | Forks: {forks})")
                        
            page += 1
            time.sleep(3)
            
    with open("repos.txt", "w") as f:
        for url in rigorous_repos: f.write(url + "\n")
    print(f"\n[SUCCESS] Extracted {len(rigorous_repos)} highly rigorous repos to repos.txt.")

if __name__ == "__main__":
    fetch_rigorous_wild_repos()