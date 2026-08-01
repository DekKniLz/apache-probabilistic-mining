import os
import re
import sys

_python_tcl_dir = os.path.join(sys.base_prefix, "tcl")
if os.path.isdir(os.path.join(_python_tcl_dir, "tcl8.6")):
    os.environ.setdefault("TCL_LIBRARY", os.path.join(_python_tcl_dir, "tcl8.6"))
if os.path.isdir(os.path.join(_python_tcl_dir, "tk8.6")):
    os.environ.setdefault("TK_LIBRARY", os.path.join(_python_tcl_dir, "tk8.6"))

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


PDS_ALGORITHM_PATTERNS = [
    ("Cuckoo Filter", ["cuckoo"]),
    ("Ribbon Filter", ["ribbon"]),
    ("XOR Filter", ["xorfilter", "xor"]),
    ("HyperLogLog", ["hyperloglog", "hll"]),
    ("LogLog", ["loglog"]),
    ("Linear Counting", ["linearcount", "linearcounting"]),
    ("CPC", ["cpc"]),
    ("Count-Min Sketch", ["countmin", "cms"]),
    ("Count Sketch", ["countsketch"]),
    ("Bloom Filter", ["bloom"]),
    ("Counter Stacks", ["counterstack", "counterstacks"]),
    ("SHARDS", ["shards"]),
]

PDS_ALGORITHM_TO_CATEGORY = {
    "Bloom Filter": "Membership Filter",
    "Ribbon Filter": "Membership Filter",
    "XOR Filter": "Membership Filter",
    "Cuckoo Filter": "Membership Filter",
    "HyperLogLog": "Cardinality Estimator",
    "LogLog": "Cardinality Estimator",
    "Linear Counting": "Cardinality Estimator",
    "CPC": "Cardinality Estimator",
    "Count-Min Sketch": "Frequency Sketch",
    "Count Sketch": "Frequency Sketch",
    "Counter Stacks": "Miss Ratio Curve Estimator",
    "SHARDS": "Miss Ratio Curve Estimator",
}

MODERN_ALGORITHMS = [
    "Cuckoo Filter", "Count-Min Sketch", "Ribbon Filter", "XOR Filter",
    "Count Sketch", "CPC", "Counter Stacks", "SHARDS",
]

KNOWN_EXTERNAL_LIBS = ['guava', 'algebird', 'clearspring', 'fastutil', 'apache.commons']


def _clean_identifier(raw_class_name):
    """Lowercases and strips all non-alphanumeric characters, so patterns
    like 'Count-Min', 'count_min' and 'CountMin' all normalize to the same
    comparable string: 'countmin'."""
    return re.sub(r'[^a-z0-9]', '', str(raw_class_name).lower())


def normalize_pds_algorithm(raw_class_name):
    """
    Identifies the concrete PDS algorithm implemented/instantiated by the
    AST node, based on the raw class name.

    Examples:
        BloomFilter, CountingBloomFilter, ScalableBloomFilter -> Bloom Filter
        HyperLogLog, HLL, HyperLogLogPlus                     -> HyperLogLog
        CountMinSketch, CMS, CountMin                          -> Count-Min Sketch
        CuckooFilter, ConcurrentCuckooFilter                   -> Cuckoo Filter

    Returns 'Unknown Algorithm' if the class name doesn't match any known
    pattern. This is intentional: it lets the pipeline surface algorithms
    that were not initially anticipated, instead of silently miscategorizing
    them.
    """
    cleaned = _clean_identifier(raw_class_name)

    for algorithm_name, patterns in PDS_ALGORITHM_PATTERNS:
        if any(pattern in cleaned for pattern in patterns):
            return algorithm_name

    return "Unknown Algorithm"


def normalize_pds_category(algorithm_name):
    """
    Maps a concrete PDS algorithm to its functional category, using a
    literature-based taxonomy (not an ad-hoc one):

        Bloom Filter, Ribbon Filter, XOR Filter, Cuckoo Filter -> Membership Filter
        HyperLogLog, LogLog, Linear Counting, CPC              -> Cardinality Estimator
        Count-Min Sketch, Count Sketch                         -> Frequency Sketch

    Returns 'Unclassified' if the algorithm has no known category yet
    (including 'Unknown Algorithm'). Extending the taxonomy with a new
    algorithm only requires adding an entry to PDS_ALGORITHM_TO_CATEGORY.
    """
    return PDS_ALGORITHM_TO_CATEGORY.get(algorithm_name, "Unclassified")


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
    """RQ3: Clasifica el origen de la implementación.

    IMPORTANTE: esta clasificación es best-effort. Para una INSTANTIATION,
    no podemos determinar con certeza si la clase proviene de una librería
    externa o de código propio sin cruzarla con el import resuelto en el
    mismo archivo (dato que miner.py no captura todavía). Por eso las
    instanciaciones se marcan como 'Unresolved' en vez de asumir que son
    internas.
    """
    context = row['context']
    target_str = str(row['target_structure']).lower()
    repo_name = str(row['repository']).lower()

    if any(lib in target_str for lib in KNOWN_EXTERNAL_LIBS):
        return 'Third-Party Library'

    if context == 'IMPORT':
        if repo_name in target_str:
            return 'Custom Implementation'
        return 'External (Unclassified)'

    if context == 'INSTANTIATION':
        return 'Unresolved (needs import cross-reference)'

    return 'Unresolved'


def enrich_dataset(df):
    """Executes the data transformation pipeline.

    Pipeline: raw_class -> pds_algorithm -> pds_category
    """
    print("[INFO] Executing Data Wrangling pipeline...")

    # 1. Clean raw AST strings
    df['clean_import'] = df['target_structure'].astype(str).str.replace('import ', '', regex=False).str.replace(';', '', regex=False).str.strip()
    df['raw_class'] = df['clean_import'].apply(lambda x: x.split('.')[-1] if '.' in x else x)

    # 2. Map dimensions for Research Questions
    df['pds_algorithm'] = df['raw_class'].apply(normalize_pds_algorithm)
    df['pds_category'] = df['pds_algorithm'].apply(normalize_pds_category)
    df['usage_domain'] = df['relative_path'].apply(infer_usage_domain)
    df['implementation_origin'] = df.apply(determine_origin, axis=1)

    # 3. Tech Generation (RQ4) -- same Legacy vs Modern logic as before,
    # now driven by pds_algorithm instead of pds_family.
    df['tech_generation'] = df['pds_algorithm'].apply(
        lambda x: 'Modern' if x in MODERN_ALGORITHMS else ('Legacy' if x != 'Unknown Algorithm' else 'Unknown')
    )

    # Filter out noise: unclassified categories and unknown algorithms.
    df = df[(df['pds_category'] != 'Unclassified') & (df['pds_algorithm'] != 'Unknown Algorithm')]

    return df


def coverage_stats(df, group_col):
    """Reporta cobertura en tres niveles para una columna dada
    (por ejemplo 'pds_algorithm' o 'pds_category'):
    - n_occurrences: total de nodos AST detectados (cuenta cruda)
    - n_files: archivos distintos donde aparece
    - n_repos: repositorios distintos donde aparece
    Esto evita que un solo repositorio grande domine las estadísticas.
    """
    return (
        df.groupby(group_col)
        .agg(
            n_occurrences=(group_col, 'size'),
            n_files=('relative_path', 'nunique'),
            n_repos=('repository', 'nunique'),
        )
        .sort_values('n_repos', ascending=False)
    )


def generate_msr_plots(df):
    """Generates publication-ready visualizations mapping to the thesis Research Questions."""
    print("[INFO] Rendering scientific visualizations...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # --- RQ1: Most Used Structures (by functional category) ---
    # NOTE: this counts every AST occurrence (import/instantiation), so a
    # single repository that uses the same structure many times will weigh
    # more than a repository that uses it once. This measures *usage
    # intensity*, not *adoption breadth* -- see RQ1b below for the latter.
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='pds_category', order=df['pds_category'].value_counts().index, hue='pds_category', palette='viridis', legend=False)
    plt.title('RQ1: Adoption of Probabilistic Data Structures in Apache Foundation Projects', fontweight='bold', pad=15)
    plt.ylabel('Frequency (Total AST Occurrences, all repos combined)')
    plt.xlabel('PDS Category')
    plt.xticks(rotation=15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ1_Adoption.png", dpi=300)
    plt.close()

    # --- RQ1b: Adoption Breadth (distinct repositories per category) ---
    # Complements RQ1: instead of raw occurrence counts, this counts how
    # many *distinct* repositories adopt each category at least once. A
    # category could rank high in RQ1 simply because one repo uses it
    # heavily, while ranking low here if few repos actually adopted it.
    if 'repository' in df.columns:
        plt.figure(figsize=(10, 6))
        repo_breadth = df.groupby('pds_category')['repository'].nunique().sort_values(ascending=False)
        sns.barplot(x=repo_breadth.index, y=repo_breadth.values, hue=repo_breadth.index, palette='viridis', legend=False)
        plt.title('RQ1b: Adoption Breadth (Distinct Repositories per PDS Category)', fontweight='bold', pad=15)
        plt.ylabel('Number of Distinct Repositories')
        plt.xlabel('PDS Category')
        plt.xticks(rotation=15)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/RQ1b_Adoption_Breadth.png", dpi=300)
        plt.close()
    else:
        print("[WARN] Column 'repository' not found; skipping RQ1b (adoption breadth) plot.")

    # --- RQ2: Usage Context ---
    plt.figure(figsize=(12, 6))
    domain_counts = df.groupby(['usage_domain', 'pds_category']).size().reset_index(name='count')
    sns.barplot(data=domain_counts, x='usage_domain', y='count', hue='pds_category', palette='Set2')
    plt.title('RQ2: Inferencia contextual basada en rutas de archivo (no validación arquitectónica)', fontweight='bold', pad=15)
    plt.figtext(0.5, -0.02, "*usage_domain se infiere de palabras clave en la ruta del archivo, no de un análisis arquitectónico validado.", ha="center", fontsize=9, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
    plt.ylabel('File Count')
    plt.xlabel('Inferred Domain')
    plt.xticks(rotation=15)
    plt.legend(title='PDS Category', loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ2_Usage_Domain.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- RQ3: Custom vs Third-Party ---
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='pds_category', hue='implementation_origin', palette='crest')
    plt.title('RQ3: Implementation Strategies (Custom Builds vs. Third-Party Libraries)', fontweight='bold', pad=15)
    plt.ylabel('Frequency')
    plt.xlabel('PDS Category')
    plt.legend(title='Origin', loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/RQ3_Implementation_Origin.png", dpi=300)
    plt.close()

    # --- RQ4: Legacy vs Modern (based on pds_algorithm) ---
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='tech_generation', order=['Legacy', 'Modern'], hue='tech_generation', palette='flare', legend=False)
    plt.title('RQ4: Distribución de algoritmos según taxonomía tecnológica definida', fontweight='bold', pad=15)
    plt.ylabel('Implementation Count')
    plt.xlabel('Technology Generation')
    plt.figtext(0.5, -0.02, "*Legacy vs. Modern classification is based on the specific PDS algorithm (pds_algorithm).", ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
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


def compute_extractor_metrics(gold_labels_path):
    """Calcula precision/recall/F1 y matriz de confusión del extractor
    contra una muestra etiquetada manualmente.

    gold_labels_path: ruta a un CSV con columnas:
    - sample_id
    - predicted_algorithm  (lo que detectó miner.py / normalize_pds_algorithm)
    - true_algorithm       (la etiqueta correcta, asignada a mano)

    Esta muestra NO se genera automáticamente; debe construirse mediante
    revisión manual estratificada (ver Fase 3 del plan de validación).
    """
    from sklearn.metrics import classification_report, confusion_matrix

    gold = pd.read_csv(gold_labels_path)
    report = classification_report(
        gold['true_algorithm'], gold['predicted_algorithm'],
        output_dict=True, zero_division=0,
    )
    labels = sorted(set(gold['true_algorithm']) | set(gold['predicted_algorithm']))
    cm = confusion_matrix(gold['true_algorithm'], gold['predicted_algorithm'], labels=labels)
    return {"report": report, "confusion_matrix": cm.tolist(), "labels": labels}


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
    coverage_by_algorithm = coverage_stats(df, 'pds_algorithm')
    coverage_by_category = coverage_stats(df, 'pds_category')
    coverage_by_algorithm.to_json(f"{os.path.dirname(JSON_OUTPUT_PATH)}/coverage_by_algorithm.json", orient='index', indent=4)
    coverage_by_category.to_json(f"{os.path.dirname(JSON_OUTPUT_PATH)}/coverage_by_category.json", orient='index', indent=4)
    print("[INFO] Coverage stats (occurrences vs files vs repos):")
    print(coverage_by_algorithm.to_string())
    generate_msr_plots(df)
    export_to_json(df)


if __name__ == "__main__":
    main()