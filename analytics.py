import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- CONFIGURATION ---
DATASET_PATH = "./output/dataset.csv"
JSON_OUTPUT_PATH = "./output/dataset_enriched.json"
PLOTS_DIR = "./plots"

def setup_environment():
    """Creates the necessary directories for output artifacts."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

def normalize_pds_family(raw_class_name):
    """
    Data Wrangling: Groups highly customized class names into their core theoretical PDS families
    to prevent chart saturation and improve statistical accuracy.
    """
    name = str(raw_class_name).lower()
    if 'bloom' in name:
        return 'Bloom Filter'
    elif 'hyperloglog' in name or 'hll' in name:
        return 'HyperLogLog'
    elif 'cuckoo' in name:
        return 'Cuckoo Filter'
    elif 'countmin' in name or 'cms' in name:
        return 'Count-Min Sketch'
    elif 'sketch' in name:
        return 'Other Sketches'
    else:
        return 'Unclassified'

def infer_usage_domain(path):
    """RQ2: Infers the architectural domain based on the file path context."""
    path = str(path).lower()
    if any(kw in path for kw in ['net', 'rpc', 'server', 'connection', 'web']):
        return 'Networking & Web'
    elif any(kw in path for kw in ['cache', 'mem', 'buffer']):
        return 'Caching Layer'
    elif any(kw in path for kw in ['io', 'storage', 'db', 'sstable', 'disk']):
        return 'Storage & Databases'
    elif any(kw in path for kw in ['analytics', 'metrics', 'stat', 'agg']):
        return 'Analytics & Telemetry'
    return 'Core Application'

def determine_origin(row):
    """RQ3: Determines if the implementation is a custom build or a third-party library."""
    if row['context'] == 'INSTANTIATION':
        return 'Internal/Native Build'
    
    import_str = str(row['target_structure']).lower()
    repo_name = str(row['repository']).lower()
    
    if repo_name in import_str:
        return 'Custom Implementation'
    if any(lib in import_str for lib in ['guava', 'algebird', 'clearspring', 'fastutil', 'apache.commons']):
        return 'Third-Party Library'
    
    return 'Standard/Other External'

def enrich_dataset(df):
    """Executes the data transformation pipeline."""
    print("[INFO] Executing Data Wrangling pipeline...")
    
    # 1. Clean raw AST strings
    df['clean_import'] = df['target_structure'].astype(str).str.replace('import ', '', regex=False).str.replace(';', '', regex=False).str.strip()
    df['raw_class'] = df['clean_import'].apply(lambda x: x.split('.')[-1] if '.' in x else x)
    
    # 2. Map dimensions for Research Questions
    df['pds_family'] = df['raw_class'].apply(normalize_pds_family)
    df['usage_domain'] = df['relative_path'].apply(infer_usage_domain)
    df['implementation_origin'] = df.apply(determine_origin, axis=1)
    
    # 3. Tech Generation (RQ4)
    modern_structures = ['Cuckoo Filter', 'Count-Min Sketch', 'Other Sketches']
    df['tech_generation'] = df['pds_family'].apply(lambda x: 'Modern' if x in modern_structures else ('Legacy' if x != 'Unclassified' else 'Unknown'))
    
    # Filter out noise
    df = df[df['pds_family'] != 'Unclassified']
    
    return df

def generate_msr_plots(df):
    """Generates publication-ready visualizations mapping to the thesis Research Questions."""
    print("[INFO] Rendering scientific visualizations...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # --- RQ1: Most Used Structures ---
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='pds_family', order=df['pds_family'].value_counts().index, hue='pds_family', palette='viridis', legend=False)
    plt.title('RQ1: Adoption of Probabilistic Data Structures in Real-World Projects', fontweight='bold', pad=15)
    plt.ylabel('Frequency (AST Occurrences)')
    plt.xlabel('PDS Family')
    plt.xticks(rotation=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ1_Adoption.png", dpi=300)
    plt.close()

    # --- RQ2: Usage Context ---
    plt.figure(figsize=(12, 6))
    domain_counts = df.groupby(['usage_domain', 'pds_family']).size().reset_index(name='count')
    sns.barplot(data=domain_counts, x='usage_domain', y='count', hue='pds_family', palette='Set2')
    plt.title('RQ2: Inferred Architectural Domain Context', fontweight='bold', pad=15)
    plt.ylabel('File Count')
    plt.xlabel('Inferred Domain')
    plt.xticks(rotation=15)
    plt.legend(title='Data Structure', loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ2_Usage_Domain.png", dpi=300)
    plt.close()

    # --- RQ3: Custom vs Third-Party ---
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='pds_family', hue='implementation_origin', palette='crest')
    plt.title('RQ3: Implementation Strategies (Custom Builds vs. Third-Party Libraries)', fontweight='bold', pad=15)
    plt.ylabel('Frequency')
    plt.xlabel('PDS Family')
    plt.legend(title='Origin', loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ3_Implementation_Origin.png", dpi=300)
    plt.close()

    # --- RQ4: Legacy vs Modern ---
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='tech_generation', order=['Legacy', 'Modern'], hue='tech_generation', palette='flare', legend=False)
    plt.title('RQ4: Technological Debt (Legacy vs. Modern Alternatives)', fontweight='bold', pad=15)
    plt.ylabel('Implementation Count')
    plt.xlabel('Technology Generation')
    plt.figtext(0.5, -0.02, "*Legacy (Bloom, HLL) vs. Modern (Cuckoo, CMS)", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ4_Tech_Debt.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Research plots successfully exported to: {PLOTS_DIR}/")

def export_to_json(df):
    """Exports the wrangled dataset to JSON for future scalability and ML pipelines."""
    print(f"[INFO] Exporting enriched dataset to: {JSON_OUTPUT_PATH}...")
    df.to_json(JSON_OUTPUT_PATH, orient='records', indent=4)
    print("[SUCCESS] JSON export completed.")

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at {DATASET_PATH}. Please run miner.py first.")
        return

    setup_environment()
    df = pd.read_csv(DATASET_PATH)
    
    if df.empty:
        print("[WARN] Dataset is empty. Aborting analysis.")
        return
        
    df = enrich_dataset(df)
    generate_msr_plots(df)
    export_to_json(df)

if __name__ == "__main__":
    main()