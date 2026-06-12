import subprocess
import sys
import logging

# Configure logging to track the process (useful for background execution)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the software repository mining pipeline.
    """
    scripts = [
        "fetch_wild_repos.py",
        "miner.py",
        "analytics.py"
    ]
    
    logging.info("Starting the complete MSR mining pipeline...")
    
    for script in scripts:
        logging.info(f"Executing {script}...")
        try:
            # Execute the script and wait for completion
            result = subprocess.run([sys.executable, script], check=True)
            logging.info(f"Successfully completed: {script}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Process {script} failed with error {e.returncode}. Halting pipeline.")
            sys.exit(1)
            
    logging.info("Pipeline finished. All data and plots are ready.")

if __name__ == "__main__":
    main()