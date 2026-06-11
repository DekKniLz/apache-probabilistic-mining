import subprocess
import sys
import logging

# Configurar logs para seguir el proceso (muy útil si se queda corriendo mientras entrenas)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Función principal que orquesta el pipeline de minería.
    """
    scripts = [
        "fetch_wild_repos.py",
        "miner.py",
        "analytics.py"
    ]
    
    logging.info("Iniciando pipeline completo de minería...")
    
    for script in scripts:
        logging.info(f"Ejecutando {script}...")
        try:
            # Ejecutamos el script y esperamos
            result = subprocess.run([sys.executable, script], check=True)
            logging.info(f"Finalizado exitosamente: {script}")
        except subprocess.CalledProcessError as e:
            logging.error(f"El proceso {script} falló con error {e.returncode}. Deteniendo pipeline.")
            sys.exit(1)
            
    logging.info("Pipeline finalizado. Todos los datos y gráficas están listos.")

if __name__ == "__main__":
    main()