import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import analytics  # reutiliza DATASET_PATH, enrich_dataset(), etc.

# --- CONFIGURATION ---
ML_PLOTS_DIR = "./plots/ml"
ML_JSON_OUTPUT_PATH = "./output/ml_analysis_summary.json"

# Features de entrada: contexto del proyecto, NO el algoritmo/categoria
# (esas son las variables que queremos explicar / predecir, no usarlas
# como insumo). 'repository' se deja fuera del feature set por defecto:
# con alta cardinalidad y pocos repos por clase, un modelo la usaria como
# atajo para "memorizar" en vez de generalizar por contexto de uso.
CONTEXT_FEATURES = ["usage_domain", "implementation_origin", "context"]

# Columnas categoricas que describen a cada cluster una vez formado.
CLUSTER_PROFILE_COLUMNS = ["pds_category", "pds_algorithm", "usage_domain", "implementation_origin"]


def setup_environment():
    """Crea los directorios de salida necesarios para este modulo."""
    if not os.path.exists(ML_PLOTS_DIR):
        os.makedirs(ML_PLOTS_DIR)
    output_dir = os.path.dirname(ML_JSON_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_enriched_dataset():
    """Carga dataset.csv y le aplica el mismo pipeline de wrangling que
    analytics.py, para no duplicar logica de normalizacion."""
    if not os.path.exists(analytics.DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset no encontrado en {analytics.DATASET_PATH}. Corre miner.py y analytics.py primero."
        )
    df = pd.read_csv(analytics.DATASET_PATH)
    if df.empty:
        raise ValueError("El dataset esta vacio.")
    return analytics.enrich_dataset(df)


def build_feature_matrix(df, feature_cols=CONTEXT_FEATURES):
    """One-hot encoding de las variables categoricas de contexto."""
    return pd.get_dummies(df[list(feature_cols)])


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
    """Ajusta PCA y devuelve (modelo, proyeccion)."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def determine_optimal_k(X, k_range):
    """
    Calcula inertia (Elbow) y Silhouette Score para cada K en k_range,
    grafica ambos, y elige automaticamente el K con mejor Silhouette
    (en vez de fijar un numero de clusters arbitrario).
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        # Silhouette no esta definido para k == n_samples ni k == 1.
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
    """Ajusta KMeans con K clusters y devuelve las etiquetas de cluster."""
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X)
    return cluster_labels, model


def profile_clusters(df, cluster_col="cluster", describe_cols=CLUSTER_PROFILE_COLUMNS):
    """
    Describe cada cluster con la moda de sus columnas categoricas
    principales, mas el tamano (n de observaciones) de cada uno.
    """
    def _mode_or_none(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else None

    summary = df.groupby(cluster_col)[list(describe_cols)].agg(_mode_or_none)
    sizes = df.groupby(cluster_col).size().rename("n_observations")
    return summary.join(sizes)


def train_classifier(df, feature_cols, target_col, plots_dir):
    """
    Entrena un RandomForestClassifier que predice `target_col`
    (pds_category o pds_algorithm) a partir del contexto del proyecto.
    Devuelve un resumen serializable (accuracy, reporte, importancias).
    """
    X = pd.get_dummies(df[list(feature_cols)])
    y = df[target_col]

    # Si alguna clase tiene muy pocas observaciones, stratify puede fallar;
    # en ese caso se hace split simple en vez de romper el pipeline.
    try:
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)

    accuracy = accuracy_score(ytest, predictions)
    report = classification_report(ytest, predictions, zero_division=0, output_dict=True)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="crest", legend=False)
    plt.title(f"Importancia de Variables -- Prediciendo {target_col}", fontweight="bold", pad=15)
    plt.xlabel("Importancia (Gini)")
    plt.ylabel("Variable (one-hot)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/Feature_Importance_{target_col}.png", dpi=300)
    plt.close()

    return {
        "target": target_col,
        "n_train": int(len(Xtrain)),
        "n_test": int(len(Xtest)),
        "n_classes": int(y.nunique()),
        "accuracy": float(accuracy),
        "classification_report": report,
    }


def main():
    setup_environment()

    try:
        df = load_enriched_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    n_rows = len(df)
    if n_rows < 10:
        print(f"[WARN] Solo hay {n_rows} filas tras el wrangling; PCA/Clustering/Clasificacion "
              f"necesitan mas datos para ser confiables. Corre el miner sobre mas repositorios.")
        return

    print(f"[INFO] Dataset enriquecido: {n_rows} observaciones.")

    # --- 1. One-Hot Encoding ---
    X = build_feature_matrix(df, CONTEXT_FEATURES)

    # --- 2. PCA ---
    print("[INFO] Ejecutando PCA...")
    pca, X_pca = run_pca(X, n_components=2)
    explained = pca.explained_variance_ratio_
    print(f"[INFO] Varianza explicada por PC1/PC2: {explained[0]:.2%} / {explained[1]:.2%}")

    plot_pca_projection(
        X_pca, df["pds_category"],
        "Proyeccion PCA por Categoria Funcional (pds_category)",
        f"{ML_PLOTS_DIR}/PCA_by_category.png",
        palette="viridis",
    )
    plot_pca_projection(
        X_pca, df["pds_algorithm"],
        "Proyeccion PCA por Algoritmo Especifico (pds_algorithm)",
        f"{ML_PLOTS_DIR}/PCA_by_algorithm.png",
        palette="tab20",
    )

    # --- 3. Eleccion de K (Elbow + Silhouette) ---
    print("[INFO] Determinando K optimo (Elbow + Silhouette)...")
    max_k = min(10, n_rows - 1)
    k_range = range(2, max_k + 1)
    best_k, inertia_by_k, silhouette_by_k = determine_optimal_k(X, k_range)
    print(f"[INFO] K elegido automaticamente por Silhouette Score: {best_k}")

    # --- 4. Clustering ---
    cluster_labels, _ = run_clustering(X, best_k)
    df = df.copy()
    df["cluster"] = cluster_labels

    plot_pca_projection(
        X_pca, df["cluster"].astype(str),
        f"Clusters Descubiertos por KMeans (K={best_k})",
        f"{ML_PLOTS_DIR}/PCA_by_cluster.png",
        palette="Set2",
    )

    cluster_profile = profile_clusters(df)
    print("[INFO] Perfil de clusters (moda de variables categoricas):")
    print(cluster_profile.to_string())

    # --- 5. Clasificacion supervisada ---
    print("[INFO] Entrenando clasificador (pds_category)...")
    clf_category = train_classifier(df, CONTEXT_FEATURES, "pds_category", ML_PLOTS_DIR)
    print(f"[INFO] Accuracy prediciendo pds_category: {clf_category['accuracy']:.2%}")

    print("[INFO] Entrenando clasificador (pds_algorithm)...")
    clf_algorithm = train_classifier(df, CONTEXT_FEATURES, "pds_algorithm", ML_PLOTS_DIR)
    print(f"[INFO] Accuracy prediciendo pds_algorithm: {clf_algorithm['accuracy']:.2%}")

    # --- 6. Exportar resumen ---
    summary = {
        "n_observations": n_rows,
        "pca_explained_variance_ratio": {"PC1": float(explained[0]), "PC2": float(explained[1])},
        "optimal_k": best_k,
        "inertia_by_k": {int(k): float(v) for k, v in inertia_by_k.items()},
        "silhouette_by_k": {int(k): float(v) for k, v in silhouette_by_k.items()},
        "cluster_profile": cluster_profile.reset_index().to_dict(orient="records"),
        "classification_pds_category": clf_category,
        "classification_pds_algorithm": clf_algorithm,
    }
    with open(ML_JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Resumen de analisis avanzado exportado a: {ML_JSON_OUTPUT_PATH}")
    print(f"[SUCCESS] Graficas exportadas a: {ML_PLOTS_DIR}/")


if __name__ == "__main__":
    main()