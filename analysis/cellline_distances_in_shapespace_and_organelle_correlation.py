"""
Analysis of Distance in Shapespace and Differences in Organelle Correlation
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

# Load configuration
sys.path.append("..")
import configs.config_all as cfg


def load_shapespace():
    """Load shapespace dataframe and map cell line IDs to names."""
    shape_mode_path = f"{cfg.PROJECT_DIR}/all_celllines/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"
    df = pd.read_csv(f"{shape_mode_path}/transformed_matrix.csv")
    df["cellline"] = df.cell_line.map(lambda x: cfg.CELL_LINE[x])
    return df


def compute_cellline_means(shapespacedf):
    """Compute mean PC coefficients per cell line and return correlation matrix."""
    cellline_mean_coefs = (
        shapespacedf.groupby("cellline")
        .agg({f"PC{i}": "mean" for i in range(1, 7)})
        .reset_index()
        .set_index("cellline")
    )
    return cellline_mean_coefs.T.corr()


def pairwise_mae_matrices(matrices):
    """
    Compute pairwise mean absolute error between square correlation matrices.

    Parameters
    ----------
    matrices : list of pd.DataFrame
        Each DataFrame is a square symmetric correlation matrix.

    Returns
    -------
    np.ndarray
        Symmetric matrix of pairwise MAEs.
    """
    n = len(matrices)
    dists = np.full((n, n), np.nan)

    for i, j in combinations(range(n), 2):
        A, B = matrices[i], matrices[j]

        # Compare only shared organelles
        common_orgs = A.columns.intersection(B.columns)
        A = A.loc[common_orgs, common_orgs].values
        B = B.loc[common_orgs, common_orgs].values

        triu_idx = np.triu_indices(A.shape[0], k=1)
        a_vals, b_vals = A[triu_idx], B[triu_idx]

        valid_mask = ~np.isnan(a_vals) & ~np.isnan(b_vals)
        dist = np.mean(np.abs(a_vals[valid_mask] - b_vals[valid_mask])) if np.any(valid_mask) else np.nan

        dists[i, j] = dists[j, i] = dist

    np.fill_diagonal(dists, 0)
    return dists


def load_organelle_matrices():
    """Load organelle correlation matrices for each cell line."""
    return [
        pd.read_csv(
            f"{cfg.PROJECT_DIR}/{cellline}/warps_protein_avg_otsu/PC1_bin3_pearsonr_df.csv",
            index_col=0,
        )
        for cellline in cfg.CELL_LINE
    ]


def plot_dendrogram_and_heatmap(matrix, labels, title_dendrogram, title_heatmap, cmap, vmin, vmax, cbar_label, ax_dendro, ax_heat):
    """Helper function to plot dendrogram + reordered heatmap side by side."""
    Z = linkage(matrix, method="average")
    dendrogram(Z, labels=labels, ax=ax_dendro)
    ax_dendro.set_title(title_dendrogram)
    for tick in ax_dendro.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    order = leaves_list(Z)
    reordered = matrix[order][:, order] if isinstance(matrix, np.ndarray) else matrix.values[order][:, order]
    reordered_labels = [labels[i] for i in order]

    sns.heatmap(
        reordered,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=reordered_labels,
        yticklabels=reordered_labels,
        cbar_kws={"label": cbar_label},
        ax=ax_heat,
    )
    ax_heat.set_title(title_heatmap)


def main():
    sns.set(style="white", context="talk")

    # Shapespace similarity
    shapespacedf = load_shapespace()
    avg_cellline = compute_cellline_means(shapespacedf)
    avg_cellline = avg_cellline.loc[cfg.CELL_LINE, cfg.CELL_LINE]

    # Organelle correlation distances
    matrices = load_organelle_matrices()
    mae_matrix = pairwise_mae_matrices(matrices)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    plot_dendrogram_and_heatmap(
        avg_cellline,
        cfg.CELL_LINE,
        "Dendrogram: Similarity in Shapespace",
        "Heatmap: Similarity in Shapespace",
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        cbar_label="Correlation",
        ax_dendro=axes[0, 0],
        ax_heat=axes[0, 1],
    )

    plot_dendrogram_and_heatmap(
        mae_matrix,
        cfg.CELL_LINE,
        "Dendrogram: Organelle Correlation Distance",
        "Heatmap: Organelle Correlation Distance",
        cmap="RdBu",
        vmin=0,
        vmax=0.45,
        cbar_label="MAE",
        ax_dendro=axes[1, 0],
        ax_heat=axes[1, 1],
    )

    plt.tight_layout()
    plt.savefig(f"{cfg.PROJECT_DIR}/shapespace_organelle_corr_dendrogram.png", dpi=300)
    plt.show()
    print(f'Figure saved to {cfg.PROJECT_DIR}/shapespace_organelle_corr_dendrogram.png')


if __name__ == "__main__":
    main()
