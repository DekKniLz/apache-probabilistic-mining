import json
import os
import sys

_python_tcl_dir = os.path.join(sys.base_prefix, "tcl")
if os.path.isdir(os.path.join(_python_tcl_dir, "tcl8.6")):
    os.environ.setdefault("TCL_LIBRARY", os.path.join(_python_tcl_dir, "tcl8.6"))
if os.path.isdir(os.path.join(_python_tcl_dir, "tk8.6")):
    os.environ.setdefault("TK_LIBRARY", os.path.join(_python_tcl_dir, "tk8.6"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, silhouette_score
from sklearn.model_selection import GroupKFold

import analytics  # reutiliza DATASET_PATH, enrich_dataset(), etc.

# --- CONFIGURATION ---
ML_PLOTS_DIR = "./plots/ml"
ML_JSON_OUTPUT_PATH = "./output/ml_analysis_summary.json"

# Features de entrada: contexto del proyecto, NO el algoritmo/categoria.
# IMPORTANTE (fuga de informacion): 'implementation_origin' se deriva en
# parte de 'context', y ambos son senales debiles. Se predice pds_category /
# pds_algorithm SOLO desde contexto estructural (ruta + tipo de nodo), nunca
# desde el nombre de la clase, para que la tarea de clasificacion sea honesta.
CONTEXT_FEATURES = ["usage_domain", "implementation_origin", "context", "path_depth", "path_has_test", "path_has_src"]

# Columnas categoricas que describen a cada cluster una vez formado.
CLUSTER_PROFILE_COLUMNS = ["pds_category", "pds_algorithm", "usage_domain", "implementation_origin"]

RANDOM_STATE = 42


def setup_environment():
    """Crea los directorios de salida necesarios para este modulo."""
    if not os.path.exists(ML_PLOTS_DIR):
        os.makedirs(ML_PLOTS_DIR)
    output_dir = os.path.dirname(ML_JSON_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_enriched_dataset():
    """Carga dataset.csv y le aplica el mismo pipeline de wrangling que analytics.py."""
    if not os.path.exists(analytics.DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset no encontrado en {analytics.DATASET_PATH}. Corre miner.py y analytics.py primero."
        )
    df = pd.read_csv(analytics.DATASET_PATH)
    if df.empty:
        raise ValueError("El dataset esta vacio.")
    return analytics.enrich_dataset(df)


def engineer_context_features(df):
    """Adds lightweight path-derived features that complement the existing context columns."""
    df = df.copy()
    path_values = df["relative_path"].fillna("").astype(str)
    df["path_depth"] = path_values.str.replace("\\", "/").str.count("/") + 1
    df["path_has_test"] = path_values.str.lower().str.contains("test|tests")
    df["path_has_src"] = path_values.str.lower().str.contains("src")
    return df


def build_feature_matrix(df, feature_cols=CONTEXT_FEATURES):
    """One-hot encoding de las variables categoricas de contexto."""
    feature_frame = df[list(feature_cols)].copy()
    return pd.get_dummies(feature_frame)


def plot_pca_projection(X_pca, labels, title, filename, palette="viridis"):
    """Grafica una proyeccion PCA de 2 componentes coloreada por `labels`."""
    plt.figure(figsize=(9, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=palette, alpha=0.8, s=70, edgecolor="white", linewidth=0.4)
    plt.title(title, fontweight="bold", pad=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def run_pca(X, n_components=2):
    """Ajusta PCA y devuelve (modelo, proyeccion).

    NOTA: PCA sobre variables one-hot (binarias) es exploratorio; la varianza
    explicada tiende a ser baja. Se usa solo para VISUALIZAR estructura, no
    como reduccion de dimensionalidad para el clasificador.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def determine_optimal_k(X, k_range):
    """Calcula inertia y Silhouette Score para cada K y elige el mejor K."""
    inertias = []
    silhouettes = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, cluster_labels))

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(list(k_range), inertias, marker="o", color="tab:blue", label="Inertia (Elbow)")
    ax1.set_xlabel("Numero de Clusters (K)")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(list(k_range), silhouettes, marker="s", color="tab:orange", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Seleccion de K: Elbow Method vs. Silhouette Score", fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig(f"{ML_PLOTS_DIR}/K_Selection_Elbow_Silhouette.png", dpi=300)
    plt.close()

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    inertia_by_k = dict(zip(k_range, inertias))
    silhouette_by_k = dict(zip(k_range, silhouettes))
    return best_k, inertia_by_k, silhouette_by_k


def run_clustering(X, k):
    """Ajusta KMeans y mete una segunda vista con clustering jerarquico."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    hierarchical = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")

    return {
        "kmeans": kmeans.fit_predict(X),
        "hierarchical": hierarchical.fit_predict(X),
    }


def detect_anomalies(X, contamination=0.05):
    """Flags unusual combinations of context features using Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    return labels == -1, model


def profile_clusters(df, cluster_col="cluster", describe_cols=CLUSTER_PROFILE_COLUMNS):
    """Describe cada cluster con la moda de sus columnas categoricas principales."""
    def _mode_or_none(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else None

    summary = df.groupby(cluster_col)[list(describe_cols)].agg(_mode_or_none)
    sizes = df.groupby(cluster_col).size().rename("n_observations")
    return summary.join(sizes)


def train_classifier(df, feature_cols, target_col, plots_dir):
    """Evalua un RandomForest con VALIDACION CRUZADA agrupada por repositorio.

    Cambio metodologico clave frente a la version anterior (un solo
    GroupShuffleSplit 80/20): se usa GroupKFold para que
    (a) ningun repositorio aparezca en train y test a la vez (evita fuga), y
    (b) se reporte media +/- desviacion estandar de las metricas, no un unico
    numero de alta varianza. El baseline mayoritario se evalua con el mismo
    esquema de folds para una comparacion justa.
    """
    X = pd.get_dummies(df[list(feature_cols)])
    y = df[target_col].reset_index(drop=True)
    groups = df["repository"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    n_groups = groups.nunique()
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return {
            "target": target_col,
            "error": f"Solo hay {n_groups} repositorio(s) con datos; se necesitan >=2 para GroupKFold.",
        }

    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = {"balanced_accuracy": [], "macro_f1": []}
    baseline_metrics = {"balanced_accuracy": []}

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        fold_metrics["balanced_accuracy"].append(balanced_accuracy_score(yte, preds))
        fold_metrics["macro_f1"].append(f1_score(yte, preds, average="macro", zero_division=0))

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(Xtr, ytr)
        baseline_metrics["balanced_accuracy"].append(balanced_accuracy_score(yte, dummy.predict(Xte)))

    ## Modelo final sobre TODOS los datos, solo para importancias y un reporte
    # de clasificacion indicativo (no para estimar generalizacion).
    final_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    final_model.fit(X, y)
    # Pass the full label set explicitly so the report is complete and does not
    # emit "classes not in y_true" warnings for rare classes.
    report_labels = sorted(y.unique())
    full_report = classification_report(
        y, final_model.predict(X), labels=report_labels, zero_division=0, output_dict=True
    )

    importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="crest", legend=False)
    plt.title(f"Importancia de Variables -- Prediciendo {target_col}", fontweight="bold", pad=15)
    plt.xlabel("Importancia (Gini)")
    plt.ylabel("Variable (one-hot)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/Feature_Importance_{target_col}.png", dpi=300)
    plt.close()

    def _mean_std(values):
        arr = np.array(values, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "per_fold": [float(v) for v in arr]}

    return {
        "target": target_col,
        "cv_scheme": f"GroupKFold(n_splits={n_splits}) grouped by repository",
        "n_observations": int(len(X)),
        "n_classes": int(y.nunique()),
        "n_groups_repositories": int(n_groups),
        "balanced_accuracy_cv": _mean_std(fold_metrics["balanced_accuracy"]),
        "macro_f1_cv": _mean_std(fold_metrics["macro_f1"]),
        "baseline_majority_balanced_accuracy_cv": _mean_std(baseline_metrics["balanced_accuracy"]),
        "in_sample_classification_report": full_report,
    }


def main():
    setup_environment()

    try:
        df = load_enriched_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    df = engineer_context_features(df)
    n_rows = len(df)
    if n_rows < 10:
        print(f"[WARN] Solo hay {n_rows} filas tras el wrangling; PCA/Clustering/Clasificacion necesitan mas datos para ser confiables.")
        return

    print(f"[INFO] Dataset enriquecido: {n_rows} observaciones.")

    X = build_feature_matrix(df, CONTEXT_FEATURES)

    print("[INFO] Ejecutando PCA...")
    pca, X_pca = run_pca(X, n_components=2)
    explained = pca.explained_variance_ratio_
    print(f"[INFO] Varianza explicada por PC1/PC2: {explained[0]:.2%} / {explained[1]:.2%}")

    plot_pca_projection(X_pca, df["pds_category"], "Proyeccion PCA por Categoria Funcional (pds_category)", f"{ML_PLOTS_DIR}/PCA_by_category.png", palette="viridis")
    plot_pca_projection(X_pca, df["pds_algorithm"], "Proyeccion PCA por Algoritmo Especifico (pds_algorithm)", f"{ML_PLOTS_DIR}/PCA_by_algorithm.png", palette="tab20")

    print("[INFO] Determinando K optimo (Elbow + Silhouette)...")
    max_k = min(10, n_rows - 1)
    k_range = range(2, max_k + 1)
    best_k, inertia_by_k, silhouette_by_k = determine_optimal_k(X, k_range)
    print(f"[INFO] K elegido automaticamente por Silhouette Score: {best_k}")

    cluster_results = run_clustering(X, best_k)
    df = df.copy()
    df["cluster"] = cluster_results["kmeans"]
    df["hierarchical_cluster"] = cluster_results["hierarchical"]

    anomaly_mask, _ = detect_anomalies(X)
    df["anomaly_flag"] = anomaly_mask.astype(int)

    plot_pca_projection(X_pca, df["cluster"].astype(str), f"Clusters Descubiertos por KMeans (K={best_k})", f"{ML_PLOTS_DIR}/PCA_by_cluster.png", palette="Set2")

    cluster_profile = profile_clusters(df)
    print("[INFO] Perfil de clusters (moda de variables categoricas):")
    print(cluster_profile.to_string())
    print(f"[INFO] Anomaly detection flagged {int(df['anomaly_flag'].sum())} observations as unusual.")

    print("[INFO] Entrenando clasificador (pds_category) con GroupKFold...")
    clf_category = train_classifier(df, CONTEXT_FEATURES, "pds_category", ML_PLOTS_DIR)
    if "balanced_accuracy_cv" in clf_category:
        ba = clf_category["balanced_accuracy_cv"]
        base = clf_category["baseline_majority_balanced_accuracy_cv"]
        print(f"[INFO] pds_category -> balanced_accuracy CV: {ba['mean']:.2%} +/- {ba['std']:.2%} "
              f"(baseline mayoritario: {base['mean']:.2%})")

    print("[INFO] Entrenando clasificador (pds_algorithm) con GroupKFold...")
    clf_algorithm = train_classifier(df, CONTEXT_FEATURES, "pds_algorithm", ML_PLOTS_DIR)
    if "balanced_accuracy_cv" in clf_algorithm:
        ba = clf_algorithm["balanced_accuracy_cv"]
        base = clf_algorithm["baseline_majority_balanced_accuracy_cv"]
        print(f"[INFO] pds_algorithm -> balanced_accuracy CV: {ba['mean']:.2%} +/- {ba['std']:.2%} "
              f"(baseline mayoritario: {base['mean']:.2%})")

    summary = {
        "n_observations": n_rows,
        "pca_explained_variance_ratio": {"PC1": float(explained[0]), "PC2": float(explained[1])},
        "optimal_k": best_k,
        "inertia_by_k": {int(k): float(v) for k, v in inertia_by_k.items()},
        "silhouette_by_k": {int(k): float(v) for k, v in silhouette_by_k.items()},
        "cluster_profile": cluster_profile.reset_index().to_dict(orient="records"),
        "n_anomalies": int(df["anomaly_flag"].sum()),
        "classification_pds_category": clf_category,
        "classification_pds_algorithm": clf_algorithm,
    }
    with open(ML_JSON_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Resumen de analisis avanzado exportado a: {ML_JSON_OUTPUT_PATH}")
    print(f"[SUCCESS] Graficas exportadas a: {ML_PLOTS_DIR}/")


if __name__ == "__main__":
    main()
