import argparse
import logging
import os
import subprocess
import sys

# Configure logging to track the process (useful for background execution)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Full pipeline in order. Each entry: (script, human label, flag that skips it).
PIPELINE = [
    ("fetch_apache_repos.py", "Extraction", "skip_fetch"),
    ("miner.py", "Mining", "skip_mine"),
    ("analytics.py", "Analytics", "skip_analytics"),
    ("ml_analytics.py", "Advanced/ML analytics", "skip_ml"),
]


def run_script(script):
    logging.info(f"Executing {script}...")
    subprocess.run([sys.executable, script], check=True)
    logging.info(f"Successfully completed: {script}")


def main():
    parser = argparse.ArgumentParser(description="Apache probabilistic-data-structure mining pipeline.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip repo extraction (reuse existing repos.txt).")
    parser.add_argument("--skip-mine", action="store_true", help="Skip cloning/mining (reuse existing output/dataset.csv).")
    parser.add_argument("--skip-analytics", action="store_true", help="Skip the analytics phase.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip the advanced/ML phase.")
    args = parser.parse_args()
    skip = vars(args)

    os.makedirs("output", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    logging.info("Starting the complete MSR mining pipeline...")
    for script, label, skip_key in PIPELINE:
        if skip.get(skip_key):
            logging.info(f"Skipping {label} phase ({script}).")
            continue
        try:
            run_script(script)
        except subprocess.CalledProcessError as e:
            logging.error(f"Process {script} failed with error {e.returncode}. Halting pipeline.")
            sys.exit(1)

    logging.info("Pipeline finished. All data and plots are ready.")


if __name__ == "__main__":
    main()
