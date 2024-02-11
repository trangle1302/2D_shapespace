import sys
sys.path.append("..")
import os
from imageio import imread
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from utils import helpers
from scipy.stats import pearsonr
import numpy as np
from organelle_heatmap import get_mask, get_average_intensities_cr, get_average_intensities_tsp, correlation
import configs.config as cfg
from scipy.stats import wasserstein_distance

def sliced_wasserstein(X, Y, num_proj=100, return_minmax = False):
    dim = X.shape[1]
    ests = []
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.randn(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir

        # compute 1d wasserstein, which is fast
        ests.append(wasserstein_distance(X_proj, Y_proj))
        
    dim = X.shape[0]
    for _ in range(num_proj):
        # sample uniformly from the unit sphere
        dir = np.random.randn(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X.T @ dir
        Y_proj = Y.T @ dir

        # compute 1d wasserstein, which is fast
        ests.append(wasserstein_distance(X_proj, Y_proj))
    if return_minmax:
        return np.mean(ests), np.max(ests), np.min(ests)
    else:
        return np.mean(ests)

def custom_heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar

def get_cells(cells_assigned, PC="PC1", bin_=[0,1,2]):
    pc_cells = cells_assigned[PC]
    ls = [pc_cells[b] for b in bin_]
    ls = helpers.flatten_list(ls)
    return ls

def plot_image_collage(paths, n_cols=5, figsize=(20, 20)):
    n_rows = len(paths) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        if i < len(paths):
            ax.imshow(imread(paths[i]))
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_pilr_collage(paths, n_cols=5, figsize=(20, 20)):
    n_rows = len(paths) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        if i < len(paths):
            ax.imshow(np.load(paths[i]))
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def get_heatmap(mappings0, cells_assigned, pc_name="PC1", pathway_group = 'Glycolysis'):
    intensity_sampling_concentric_ring = False
    intensity_warping = True
    project_dir = cfg.PROJECT_DIR
    if intensity_sampling_concentric_ring:
        sampled_intensity_dir = f"{project_dir}/sampled_intensity_bin"
    if intensity_warping:
        sampled_intensity_dir = f"{project_dir}/warps" 
    mappings_ = mappings0[mappings0.sc_target != "Negative"]
    #mappings_ = mappings0[mappings0.PathwayGroup == pathway_group]
    mappings_ = mappings0[mappings0.Pathway == pathway_group]
    merged_bins = [[0,1,2], [3], [4,5,6]]
    shape_ = (31, 256) if intensity_sampling_concentric_ring else (336, 700)
    keep_orgs = mappings_.locations.unique()
    #keep_orgs = [o for o in keep_orgs if o in cfg.ORGANELLES]
    if intensity_sampling_concentric_ring:
        shape_= (31, 256)
    else:
        shape_ = (336, 700) #(179, 317)
    lines = []
    lines.append(["PC","PathwayGroup", "bin","n_cells"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
    for i, bin_ in enumerate(merged_bins):
        images = {}
        for org in keep_orgs: #cfg.ORGANELLES:
            ls = get_cells(cells_assigned, PC=pc_name, bin_=bin_)
            df_sl = mappings_[mappings_.cell_idx.isin([os.path.basename(l).replace(".npy", "") for l in ls])]
            #ls_ = df_sl[df_sl.locations.str.contains(org)].cell_idx.to_list()
            ls_ = df_sl[df_sl.sc_target == org].cell_idx.to_list()
            n0 = len(ls_)
            lines.append([pc_name, org, bin_[0], n0])         

            if len(ls_)==0 or len(ls_) < 3:
                intensities = np.zeros(shape_) 
            else:      
                if intensity_sampling_concentric_ring:
                    intensities = get_average_intensities_cr(ls_, sampled_intensity_dir=sampled_intensity_dir)
                    #print(intensities.shape)
                    #np.save(f"{avg_organelle_dir}/PC{PC}_{org}_b{bin_[0]}.npy", intensities)
                if intensity_warping:
                    intensities = get_average_intensities_tsp(ls_, sampled_intensity_dir=sampled_intensity_dir)
                    intensities = (intensities*255).astype('uint8')
                    #imwrite(f"{avg_organelle_dir}/PC{PC}_{org}_b{bin_[0]}.png", intensities)
            images[org] = intensities
        # Filter for 
        #ssim_scores = correlation(images, pearsonr, mask=None) #structural_similarity)
        ssim_scores = correlation(images, sliced_wasserstein, mask=None) #structural_similarity)
        ssim_df = pd.DataFrame(ssim_scores, columns=list(images.keys()))
        ssim_df.index = list(images.keys())
        #print(ssim_df)
        #ssim_df.to_csv(f"{avg_organelle_dir}/PC{PC}_bin{b}_pearsonr_df.csv")
        #custom_heatmap(ssim_scores, row_labels=list(images.keys()), col_labels=list(images.keys()), ax=axes[i], cmap="RdBu", vmin=-1, vmax=1)
        #axes[i].imshow(ssim_df, cmap="RdBu", vmin=-1, vmax=1)
        print(ssim_df.max(), ssim_df.min())
        sns.heatmap(ssim_df, cmap="plasma", vmin=0, vmax = 100, ax=axes[i])
        #sns.clustermap(ssim_df.fillna(0), method="complete", cmap="RdBu", vmin=-1, vmax=1, ax=axes[i])
        
    plt.yticks(rotation=30) 
    plt.tight_layout()
    fig.suptitle(f"{pc_name} - {pathway_group}")
    df = pd.DataFrame(lines[1:], columns=lines[0])
    return df