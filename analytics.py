import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH, PLOTS_DIR = "./output/dataset.csv", "./plots"
PDS_GENERATIONS = {'BloomFilter': 'Legacy', 'HyperLogLog': 'Legacy', 'IFilter': 'Legacy', 'CountMinSketch': 'Modern', 'CuckooFilter': 'Modern', 'HyperLogLogPlus': 'Modern'}

def enrich_dataset(df):
    df['clean_import'] = df['target_structure'].astype(str).str.replace('import ', '', regex=False).str.replace(';', '', regex=False).str.strip()
    df['pds_class'] = df['clean_import'].apply(lambda x: x.split('.')[-1] if '.' in x else x)
    
    def get_origin(row):
        if row['context'] == 'INSTANTIATION': return 'Internal/Unknown'
        imp = row['clean_import'].lower()
        if str(row['repository']).lower() in imp: return 'Custom (Internal)'
        if any(lib in imp for lib in ['guava', 'clearspring', 'algebird', 'fastutil']): return 'Third-Party Lib'
        return 'Standard/Other'
        
    df['origin'] = df.apply(get_origin, axis=1)
    df['generation'] = df['pds_class'].map(lambda x: PDS_GENERATIONS.get(x, 'Unclassified'))
    return df

def generate_plots(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="ticks", context="paper")

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='pds_class', order=df['pds_class'].value_counts().index, palette='mako')
    plt.title('RQ1: Adoption Frequency of PDS'); plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ1_Adoption.png", dpi=300); plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='generation', palette='flare', order=['Legacy', 'Modern'])
    plt.title('RQ4: Legacy vs Modern PDS Adoption'); plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ4_TechDebt.png", dpi=300); plt.close()
    
    print(f"[SUCCESS] MSR-ready plots generated in: {PLOTS_DIR}/")

if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    if not df.empty:
        generate_plots(enrich_dataset(df))