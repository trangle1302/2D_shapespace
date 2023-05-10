import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import json
import numpy as np
from collections import Counter
import sys

sys.path.append("..")
from utils import helpers
import statsmodels.api as sm
from scipy import stats


def get_pc_cell_assignment(cells_assigned, PC="PC1"):
    pc_cells = cells_assigned[PC]
    files = helpers.flatten_list(pc_cells)
    bins = [np.repeat(i, len(b)) for i, b in enumerate(pc_cells)]
    bins = helpers.flatten_list(bins)
    d = {"filename": files, PC: bins}
    df = pd.DataFrame.from_dict(d)
    return df


def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], "pearson")
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter(
        [0.5],
        [0.5],
        marker_size,
        [corr_r],
        alpha=0.6,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        transform=ax.transAxes,
    )
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(
        corr_text,
        [0.5, 0.5,],
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=font_size,
    )


def main():
    project_dir = f"/data/2Dshapespace/S-BIAD34"
    sc_stats = pd.read_csv(f"{project_dir}/single_cell_statistics.csv")
    print(sc_stats.columns)
    alignment = "fft_cell_major_axis_polarized"
    shape_mode_path = f"{project_dir}/shapemode/{alignment}_cell_nuclei_nux4"
    f = open(f"{shape_mode_path}/cells_assigned_to_pc_bins.json", "r")
    cells_assigned = json.load(f)
    print(cells_assigned.keys())
    for PC in cells_assigned.keys():
        df_ = get_pc_cell_assignment(cells_assigned, PC)
        df_["ab_id"] = [f.split("/")[-2] for f in df_.filename]
        df_["cell_id"] = [f.split("/")[-1].replace(".npy", "") for f in df_.filename]
        sc_stats = sc_stats.merge(
            df_[[PC, "ab_id", "cell_id"]], on=["ab_id", "cell_id"]
        )
    sc_stats.to_csv(f"{project_dir}/single_cell_statistics_pcs.csv")
    tmp = sc_stats.corr(method="pearson")
    tmp.to_csv(f"{project_dir}/single_cell_statistics_corr.csv")
    # Pseudotime correlation
    for PC in cells_assigned.keys():
        res = stats.linregress(sc_stats[PC], sc_stats.pseudotime)
        print(f"{PC}: R-squared: {res.rvalue**2:.6f}")
        gamma_model = sm.GLM(
            sc_stats[PC],
            sc_stats[["pseudotime", "cell_area"]],
            family=sm.families.Gamma(),
        ).fit()
        print(gamma_model.df_resid, gamma_model.pvalues)
        print(sc_stats[[PC, "pseudotime"]].groupby(PC).describe())

    sc_stats["MT_cell_mean"] = sc_stats["MT_cell_sum"] / sc_stats.cell_area
    sc_stats["Protein_nu_mean"] = sc_stats["Protein_nu_sum"] / sc_stats.nu_area
    sc_stats["Protein_cytosol_mean"] = (
        sc_stats["Protein_cell_sum"] - sc_stats["Protein_nu_sum"]
    ) / (sc_stats.cell_area - sc_stats.nu_area)
    tmp = sc_stats[
        [
            "cell_area",
            "nu_area",
            "nu_eccentricity",
            "pseudotime",
            "MT_cell_mean",
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "PC5",
            "PC6",
        ]
    ]
    sb.set(style="white", font_scale=1.6)
    g = sb.PairGrid(tmp, aspect=1.4, diag_sharey=False)
    g.map_lower(sb.regplot, lowess=True, ci=False, line_kws={"color": "black"})
    g.map_diag(sb.distplot, kde_kws={"color": "black"})
    g.map_upper(corrdot)
    g.savefig(f"{project_dir}/single_cell_statistics.png")

    # Meta data from the HPA, Antibody
    # ifimages = pd.read_csv("/data/kaggle-dataset/publicHPA_umap/ifimages_U2OS.csv")
    ifimages = pd.read_csv("/data/HPA-IF-images/IF-image.csv")
    ifimages = ifimages[ifimages.atlas_name == "U-2 OS"]  # filtered U2OS
    ifimages = ifimages[
        ["ensembl_ids", "gene_names", "antibody", "locations", "Ab state"]
    ].drop_duplicates()
    ifimages = ifimages[ifimages["Ab state"].isin(["IF_FINISHED", "IF_PUBLISHED"])]
    print("Number of ab: ", ifimages.antibody.nunique())

    ab_list = list(set(sc_stats[sc_stats.ab_id.isin(ifimages.antibody)].ab_id))
    print(f"Number of ab passed QC: {len(ab_list)}/{sc_stats.ab_id.nunique()}")

    keep_cols = [
        "cell_area",
        "nu_area",
        "nu_eccentricity",
        "pseudotime",
        "Protein_nu_mean",
        "Protein_cytosol_mean",
        "MT_cell_mean",
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PC6",
    ]
    for ab_id in ab_list:
        ab_df_ = sc_stats[sc_stats.ab_id == ab_id]
        ab_df_ = ab_df_[keep_cols]
        plt.figure()
        sb.set(style="white", font_scale=1.6)
        g = sb.PairGrid(ab_df_, aspect=1.4, diag_sharey=False)
        g.map_lower(sb.regplot, lowess=True, ci=False, line_kws={"color": "black"})
        g.map_diag(sb.displot, kde_kws={"color": "black"})
        g.map_upper(corrdot)
        plt.savefig(f"{project_dir}/{ab_id}")


if __name__ == "__main__":
    main()
