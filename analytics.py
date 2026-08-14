import os
import re
import sys

_python_tcl_dir = os.path.join(sys.base_prefix, "tcl")
if os.path.isdir(os.path.join(_python_tcl_dir, "tcl8.6")):
    os.environ.setdefault("TCL_LIBRARY", os.path.join(_python_tcl_dir, "tcl8.6"))
if os.path.isdir(os.path.join(_python_tcl_dir, "tk8.6")):
    os.environ.setdefault("TK_LIBRARY", os.path.join(_python_tcl_dir, "tk8.6"))

# Use a non-interactive backend so the pipeline runs headless (CI / servers).
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- CONFIGURATION ---
DATASET_PATH = "./output/dataset.csv"
JSON_OUTPUT_PATH = "./output/dataset_enriched.json"
FUNNEL_OUTPUT_PATH = "./output/classification_funnel.json"
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

# --- RQ2 domain inference vocabulary -------------------------------------
# NOTE: matching is done on WHOLE path tokens (segments split on separators
# and camelCase), NOT raw substrings. The previous substring approach caused
# systematic false positives, e.g. "Region.java" -> 'io' match, "Member.java"
# -> 'mem' match, "planet/" -> 'net' match. Token matching removes those.
DOMAIN_RULES = [
    ("Networking & Web", ["net", "rpc", "server", "connection", "web", "http", "netty", "transport"]),
    ("Caching Layer", ["cache", "buffer", "memtable", "memstore"]),
    ("Storage & Databases", ["io", "storage", "db", "sstable", "disk", "wal", "lsm", "store"]),
    ("Analytics & Telemetry", ["analytics", "metrics", "metric", "stats", "stat", "agg", "telemetry"]),
]


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


def _path_tokens(path):
    """Splits a file path into a set of lowercase word tokens.

    Splits on path separators and non-alphanumeric characters, and also
    extracts alphabetic runs so 'MemTableStore' -> {mem, table, store} via
    the lowercased segments. Used by infer_usage_domain for exact-token
    matching instead of fragile substring matching.
    """
    lowered = str(path).lower()
    parts = re.split(r'[\\/]+', lowered)
    tokens = set()
    for part in parts:
        part = re.sub(r'\.java$', '', part)
        for tok in re.split(r'[^a-z0-9]+', part):
            if tok:
                tokens.add(tok)
    return tokens


def infer_usage_domain(path):
    """RQ2: Infers the architectural domain from whole path tokens.

    This is a best-effort heuristic based on directory/file naming
    conventions, NOT a validated architectural analysis. See the Threats to
    Validity section of the paper and the caption on the RQ2 figure.
    """
    tokens = _path_tokens(path)
    for domain, keywords in DOMAIN_RULES:
        if tokens & set(keywords):
            return domain
    return 'Core Application'


def determine_origin(row):
    """RQ3: Clasifica el origen de la implementacion.

    IMPORTANTE: esta clasificacion es best-effort. Para una INSTANTIATION,
    no podemos determinar con certeza si la clase proviene de una libreria
    externa o de codigo propio sin cruzarla con el import resuelto en el
    mismo archivo (dato que miner.py no captura todavia). Por eso las
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


def annotate_dataset(df):
    """Adds all derived analysis columns WITHOUT dropping any rows.

    Kept separate from filtering so the caller can report the classification
    funnel (how many raw findings survive to the analyzed set).
    """
    df = df.copy()

    # 1. Clean raw AST strings
    df['clean_import'] = (
        df['target_structure'].astype(str)
        .str.replace('import ', '', regex=False)
        .str.replace(';', '', regex=False)
        .str.strip()
    )
    df['raw_class'] = df['clean_import'].apply(lambda x: x.split('.')[-1] if '.' in x else x)

    # 2. Map dimensions for Research Questions
    df['pds_algorithm'] = df['raw_class'].apply(normalize_pds_algorithm)
    df['pds_category'] = df['pds_algorithm'].apply(normalize_pds_category)
    df['usage_domain'] = df['relative_path'].apply(infer_usage_domain)
    df['implementation_origin'] = df.apply(determine_origin, axis=1)

    # 3. Tech Generation (RQ4): Legacy vs Modern, driven by pds_algorithm.
    df['tech_generation'] = df['pds_algorithm'].apply(
        lambda x: 'Modern' if x in MODERN_ALGORITHMS else ('Legacy' if x != 'Unknown Algorithm' else 'Unknown')
    )
    return df


def filter_classified(df):
    """Drops noise: unclassified categories and unknown algorithms."""
    mask = (df['pds_category'] != 'Unclassified') & (df['pds_algorithm'] != 'Unknown Algorithm')
    return df[mask].copy()


def enrich_dataset(df):
    """Executes the full data transformation pipeline (annotate + filter).

    Pipeline: raw_class -> pds_algorithm -> pds_category, then drop rows that
    could not be classified. Returns the analyzed DataFrame.

    NOTE: kept as a single entry point so downstream consumers (e.g.
    ml_analytics.py) get the same filtered dataset. For the classification
    funnel report, call annotate_dataset() + filter_classified() directly.
    """
    print("[INFO] Executing Data Wrangling pipeline...")
    return filter_classified(annotate_dataset(df))


def compute_classification_funnel(raw_df, annotated_df, classified_df):
    """Builds the extraction -> classification funnel for the paper.

    Reports how many raw AST findings were captured, how many were mapped to
    a known algorithm/category, and how many were dropped as unknown. This is
    the number reviewers will ask for.
    """
    n_raw = int(len(raw_df))
    n_unknown_algo = int((annotated_df['pds_algorithm'] == 'Unknown Algorithm').sum())
    n_classified = int(len(classified_df))
    funnel = {
        "raw_findings": n_raw,
        "unknown_algorithm": n_unknown_algo,
        "classified_findings": n_classified,
        "retention_rate": round(n_classified / n_raw, 4) if n_raw else 0.0,
        "by_context": annotated_df['context'].value_counts().to_dict(),
        "classified_by_category": classified_df['pds_category'].value_counts().to_dict(),
        "classified_by_algorithm": classified_df['pds_algorithm'].value_counts().to_dict(),
    }
    return funnel


def coverage_stats(df, group_col):
    """Reporta cobertura en tres niveles para una columna dada
    (por ejemplo 'pds_algorithm' o 'pds_category'):
    - n_occurrences: total de nodos AST detectados (cuenta cruda)
    - n_files: archivos distintos donde aparece
    - n_repos: repositorios distintos donde aparece
    Esto evita que un solo repositorio grande domine las estadisticas.
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
    plt.title('RQ2: Inferencia contextual basada en rutas de archivo (no validacion arquitectonica)', fontweight='bold', pad=15)
    plt.figtext(0.5, -0.02, "*usage_domain se infiere de tokens de la ruta del archivo, no de un analisis arquitectonico validado.", ha="center", fontsize=9, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
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
    plt.title('RQ4: Distribucion de algoritmos segun taxonomia tecnologica definida', fontweight='bold', pad=15)
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
    """Calcula precision/recall/F1 y matriz de confusion del extractor
    contra una muestra etiquetada manualmente.

    gold_labels_path: ruta a un CSV con columnas:
    - sample_id
    - predicted_algorithm  (lo que detecto miner.py / normalize_pds_algorithm)
    - true_algorithm       (la etiqueta correcta, asignada a mano)

    Esta muestra NO se genera automaticamente; debe construirse mediante
    revision manual estratificada (ver build_gold_sample_template).
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


def build_gold_sample_template(df, n_per_algorithm=15, seed=42, out_path="./output/gold_sample_template.csv"):
    """Draws a STRATIFIED random sample of findings for manual labeling.

    Produces a CSV you fill in by hand: for each sampled finding it records
    the predicted_algorithm and leaves true_algorithm blank. Once labeled,
    feed it to compute_extractor_metrics() to report the extractor's
    precision/recall/F1 -- the validation reviewers will expect.
    """
    frames = []
    for algo, group in df.groupby('pds_algorithm'):
        take = min(n_per_algorithm, len(group))
        frames.append(group.sample(take, random_state=seed))
    sample = pd.concat(frames).reset_index(drop=True)
    sample = sample.assign(
        sample_id=range(1, len(sample) + 1),
        predicted_algorithm=sample['pds_algorithm'],
        true_algorithm="",  # to be filled in manually
    )
    cols = ['sample_id', 'repository', 'relative_path', 'line_number',
            'context', 'target_structure', 'predicted_algorithm', 'true_algorithm']
    cols = [c for c in cols if c in sample.columns]
    sample[cols].to_csv(out_path, index=False)
    print(f"[INFO] Gold-sample template ({len(sample)} rows) written to {out_path}. Fill in 'true_algorithm' by hand.")
    return out_path


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at {DATASET_PATH}. Please run miner.py first.")
        return

    setup_environment()
    raw_df = pd.read_csv(DATASET_PATH)

    if raw_df.empty:
        print("[WARN] Dataset is empty. Aborting analysis.")
        return

    # Annotate (no rows dropped) so we can report the funnel, then filter.
    annotated = annotate_dataset(raw_df)
    df = filter_classified(annotated)

    funnel = compute_classification_funnel(raw_df, annotated, df)
    with open(FUNNEL_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(funnel, handle, indent=4, ensure_ascii=False)
    print(f"[INFO] Classification funnel: {funnel['raw_findings']} raw -> "
          f"{funnel['classified_findings']} classified "
          f"({funnel['retention_rate']:.1%} retained, "
          f"{funnel['unknown_algorithm']} unknown dropped).")

    if df.empty:
        print("[WARN] No findings survived classification. Aborting plots.")
        return

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
