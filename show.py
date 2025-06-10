import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summarize_and_plot_clustering(clustObj, out_dir="plots"):
    """
    Nicely summarize, visualize clustering results and save plots + CSV with cluster thresholds.

    Args:
        clustObj: ClustObj instanz von gene_expr_dist_hierarchy().
        out_dir (str): Verzeichnis, in dem die Plots und CSV abgelegt werden.
    """
    # Sicherstellen, dass das Ausgabe-Verzeichnis existiert
    os.makedirs(out_dir, exist_ok=True)

    # 1) Zusammenfassungstabelle erstellen
    df_summary = pd.DataFrame({
        'enzyme': clustObj.objects,
        'cluster': clustObj.cindex,
        'num_tissues_expressed': clustObj.numObjs
    })

    # 2) Cluster-Größen als Balkendiagramm
    cluster_sizes = df_summary['cluster'].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    cluster_sizes.plot(kind='bar')
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Enzymes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_sizes.png"))
    plt.close()

    # 3) Heatmap der Cluster-Zentroiden (Histogramm-Merkmale)
    centroids = clustObj.C  # Form (k, n_bins)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(centroids, aspect='auto')
    plt.colorbar(label='Normalized Frequency')
    plt.title("Cluster Centroids (Histogram Features)")
    plt.xlabel("Bin Index")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_centroids.png"))
    plt.close()

    # 4) Liste der Beispiel-Enzyme pro Cluster
    with open(os.path.join(out_dir, "cluster_examples.txt"), "w") as f:
        for ci in sorted(df_summary['cluster'].unique()):
            members = df_summary[df_summary['cluster'] == ci]['enzyme'].tolist()
            header = f"Cluster {ci} ({len(members)} enzymes) Beispiel:\n"
            f.write(header)
            for enzyme in members[:5]:
                f.write(f"  - {enzyme}\n")
            f.write("\n")

    # 5) Schwellenwerte nach StanDep-Ansatz berechnen und in CSV speichern
    V = clustObj.Data  # (enzymes x tissues) log10-transformed
    labels = np.array(clustObj.cindex)
    cluster_ids = np.unique(labels)

    # Globale Statistiken
    all_vals = V.flatten()
    M = np.nanmean(all_vals)
    D = np.nanstd(all_vals)

    # Cluster-spezifische Mittel und Std
    m_list, s_list = [], []
    for ci in cluster_ids:
        mask = (labels == ci)
        vals_c = V[mask, :].flatten()
        vals_c = vals_c[np.isfinite(vals_c)]
        m_list.append(np.nanmean(vals_c))
        s_list.append(np.nanstd(vals_c))
    m_arr = np.array(m_list)
    s_arr = np.array(s_list)

    # f(s_c) und g(m_c)
    delta_s = s_arr - D
    max_delta = np.max(delta_s)
    f_norm = delta_s / max_delta if max_delta != 0 else delta_s
    g = -(m_arr - M)
    y = f_norm + g

    # Percentile-Werte Theta_c
    y_min, y_max = np.min(y), np.max(y)
    Theta = ((y - y_min) * 100.0 / (y_max - y_min)) if y_max != y_min else np.zeros_like(y)

    # Numerische Schwellen pro Cluster
    T_vals = []
    for i, ci in enumerate(cluster_ids):
        mask = (labels == ci)
        vals_c = V[mask, :].flatten()
        vals_c = vals_c[np.isfinite(vals_c)]
        pct = Theta[i]
        threshold = np.percentile(vals_c, pct)
        T_vals.append(threshold)

    # Metadaten-DataFrame
    df_meta = pd.DataFrame({
        'cluster': cluster_ids,
        'size': [clustObj.numObjInClust[ci-1] for ci in cluster_ids],
        'mean_log_expr': m_arr,
        'std_log_expr': s_arr,
        'y_score': y,
        'theta_percentile': Theta,
        'threshold_value': T_vals
    })
    df_meta.to_csv(os.path.join(out_dir, "cluster_thresholds.csv"), index=False)

    print(f"Plots, Beispiele und Schwellenwerte wurden gespeichert in: {out_dir}/")
