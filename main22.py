import subprocess
import sys

def run_script(script_name):
    print(f"\n[ORCHESTRATOR] Starting: {script_name}...")
    try:
        # Ejecuta el script y espera a que termine
        subprocess.run([sys.executable, script_name], check=True)
        print(f"[ORCHESTRATOR] Successfully completed: {script_name}")
    except subprocess.CalledProcessError:
        print(f"[ORCHESTRATOR] ERROR in {script_name}. Stopping pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    print("--- STARTING FULL MSR MINING PIPELINE ---")
    
    # 1. Buscar repositorios "In the wild"
    run_script("fetch_wild_repos.py")
    
    # 2. Minar los datos AST
    run_script("miner.py")
    
    # 3. Analizar resultados y generar gráficas
    run_script("analytics.py")
    
    print("\n--- PIPELINE FINISHED SUCCESSFULLY ---")