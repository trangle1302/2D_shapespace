import os
from coefficients import alignment, coefs
from warps import parameterize
os.chdir("C:/Users/trang.le/Desktop/2D_shape_space")
import numpy as np
from imageio import imwrite
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
from utils import plotting
from skimage.morphology import dilation, square, erosion
import pandas as pd
import seaborn as sns

LABEL_TO_ALIAS = {
  0: 'Nucleoplasm',
  1: 'NuclearM',
  2: 'Nucleoli',
  3: 'NucleoliFC',
  4: 'NuclearS',
  5: 'NuclearB',
  6: 'EndoplasmicR',
  7: 'GolgiA',
  8: 'IntermediateF',
  9: 'ActinF',
  10: 'Microtubules',
  11: 'MitoticS',
  12: 'Centrosome',
  13: 'PlasmaM',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'VesiclesPCP',
  #18: 'Negative',
  #19:'Multi-Location',
}

all_locations = dict((v, k) for k,v in LABEL_TO_ALIAS.items())


COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]

COLORS_MAP = {
    "Nucleoplasm":"Blues",
    "GolgiA":"Greens",
    "IntermediateF":"Oranges",
    "Mitochondria":"Reds",
    "PlasmaM":"Purples",
    "Cytosol":"Greys"
    }

def open_gif(gif_path):
    animated_gif = Image.open(gif_path)#f"{organelle_dir}/{org}_PC1.gif")
    frames = [f for f in ImageSequence.Iterator(animated_gif)]
    frames = []
    for frame in ImageSequence.Iterator(animated_gif):
        fr = frame.copy()
        fr.past()

def coordinates_to_image(x, y, intensity, binarize=False, shift_x = None, shift_y = None):
    """

    Parameters
    ----------
    x : np.array shape (n_rings, n_positions)
        X coordiates of sampled points, each ring from nu_centroid-nu_membrane-cell_membrane has n_positions of x
    y : list of np.array shape (n_rings, n_positions)
        y coordiates of sampled points, each ring from nu_centroid-nu_membrane-cell_membrane has n_positions of y
    intensity : sampled intensity at (n_rings, n_positions) points
        DESCRIPTION.
    binarize : whether to binarize intensity or not
        binarize threshold = mean (TODO: udpate this to histogram normalization or other)
    shift_x : distance to add to all x coordinates
        If None, shift_y=min(x).
    shift_y : distance to add to all y coordinates
        If None, shift_y=min(y).

    Returns
    -------
    img : np.array
        output image.

    """
    assert x.shape == y.shape == intensity.shape
    
    x = np.array(np.round(x), dtype=int)
    y = np.array(np.round(y), dtype=int)
    
    if binarize:
        thres = intensity.mean()
        intensity = intensity > thres
    
    if shift_x == None:
        shift_x = x.min()
    if shift_y == None:
        shift_y = y.min()
    x = x - shift_x
    y = y - shift_y
    
    img = np.zeros((x.max()-x.min() + 1, y.max()-y.min() +1), dtype='float64')
    prev_x = []
    prev_y = []
    prev_int = []
    for i_, (ix, iy, intensity_ring) in enumerate(zip(x,y, intensity)):
        for ix_, iy_, intensity_ in zip(ix,iy, intensity_ring):
            img[ix_-5:ix_+5, iy_-5:iy_+5] = intensity_
        if i_ > 9:
            for ix_, iy_, intensity_ in zip((ix+prev_x)/2,(iy+prev_y)/2, (intensity_ring+prev_int)/2):
                img[int(ix_)-5:int(ix_)+5, int(iy_)-5:int(iy_)+5] = intensity_
        prev_x = ix
        prev_y = iy
        prev_int = intensity_ring
            
    return img

def main(plot=False):
    shape_var_dir = "C:/Users/trang.le/Desktop/shapemode/U-2_OS/PCA_ratio8"
    organelle_dir = "C:/Users/trang.le/Desktop/shapemode/organelle"
    save_dir = "C:/Users/trang.le/Desktop/shapemode/avg_cell"
    avg_coords = np.load(f"{shape_var_dir}/Avg_cell.npz")
    print(avg_coords.files)
    ix_n = avg_coords["ix_n"] #[avg_coords["ix_n"][i] for i in range(0,1280,5)]
    iy_n = avg_coords["iy_n"] #[avg_coords["iy_n"][i] for i in range(0,1280,5)]
    ix_c = avg_coords["ix_c"] #[avg_coords["ix_c"][i] for i in range(0,1280,5)]
    iy_c = avg_coords["iy_c"] #[avg_coords["iy_c"][i] for i in range(0,1280,5)]
    x_,y_ = parameterize.get_coordinates(
                np.concatenate([ix_n, iy_n]), 
                np.concatenate([ix_c, iy_c]), 
                [0,0], 
                n_isos = [10,10], 
                plot=False)
    
    avg_organelle_intensity = []
    norm = plt.Normalize(vmin=0, vmax=1)
    for org in all_locations.keys():
        intensities = np.load(f"{organelle_dir}/{org}_PC1_intensity.npy")       
        intensities = intensities[4:6].sum(axis=0) #avg 2 slice in the middle
        avg_organelle_intensity += [[org] + intensities.flatten().tolist()]
        img = coordinates_to_image(np.asarray(x_), np.asarray(y_), intensities)
        imwrite(f"{save_dir}/{org}.png",img)
        #img = coordinates_to_image(np.asarray(x_), np.asarray(y_), intensities, binarize=True)
        plt.imshow(dilation(erosion(img), square(7)))
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(f"{save_dir}/{org}.jpg", bbox_inches="tight")
        plt.close()
    avg_organelle_intensity = pd.DataFrame(avg_organelle_intensity)
    avg_organelle_intensity.index = avg_organelle_intensity.iloc[:,0]
    avg_organelle_intensity.drop([0], axis=1, inplace=True)
    # No mitotic spindle in the average cells (which makes sense!) so remove the org
    covar_mat = avg_organelle_intensity.transpose().drop(["MitoticS"],axis=1).corr()
    sns.heatmap(covar_mat, cmap="RdBu", vmin=-1, vmax=1)
    sns.clustermap(covar_mat, method="complete", cmap='RdBu', annot=True, 
               annot_kws={"size": 12}, vmin=-1, vmax=1, figsize=(15,15))
    
    if plot:
        for org,org_color in COLORS_MAP.items():
            intensities = np.load(f"{organelle_dir}/{org}_PC1_intensity.npy")       
            intensities = intensities[4:6].sum(axis=0) #avg 2 slice in the middle
            fig, ax = plt.subplots()
            for i,(xi,yi,intensity) in enumerate(zip(x_,y_,intensities)):
                ax.scatter(xi, yi,c=intensity, norm=norm, cmap= org_color)
            ax.axis("scaled")
            fig.tight_layout()
            ax.axis("off")
            ax.figure.savefig(f"{save_dir}/{org}.jpg", bbox_inches="tight")

def sample_covar_matrix(mat1, mat2):
    """

    Parameters
    ----------
    mat1 : TYPE
        DESCRIPTION.
    mat2 : TYPE
        DESCRIPTION.

    Returns
    -------
    corr : 

    """
    corr = np.cov(mat1)
    return corr

def investigate_organell_pc_var():
    """
    Sam

    Returns
    -------
    None.

    """
    shape_var_dir = "C:/Users/trang.le/Desktop/shapemode/U-2_OS/0"
    organelle_dir = "C:/Users/trang.le/Desktop/shapemode/organelle"
    save_dir = "C:/Users/trang.le/Desktop/shapemode/avg_cell"
    
    # Plot all PCs for each organelle
    for org in all_locations.keys():
        fig,ax = plt.subplots(nrows=2, ncols=6)
        for i in range(1,13):
            intensities = np.load(f"{organelle_dir}/{org}_PC{i}_intensity.npy")  
            #coords = np.load(f"{organelle_dir}/{org}_PC{i}.npz")

            ax[(i-1) //6, (i-1) % 6].imshow(intensities.mean(axis=2).T)
            #plt.xticks(["nu_centroid","","","","","","","","","","nucleus","","","","","","","","","cell"])
            #X = np.zeros((21,10))
            ax[(i-1) //6, (i-1) % 6].set_title(f"PC{i}")
        fig.suptitle(org)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/allPC_{org}.png", bbox_inches="tight")
    
    # Plot all organelles for each PC
    for i in range(1,13):
        fig,ax = plt.subplots(nrows=3, ncols=6, figsize=(25,20))
        for org, k in all_locations.items():
            intensities = np.load(f"{organelle_dir}/{org}_PC{i}_intensity.npy")  
            #coords = np.load(f"{organelle_dir}/{org}_PC{i}.npz")

            ax[k //6, k % 6].imshow(intensities.mean(axis=2).T)
            #plt.xticks(["nu_centroid","","","","","","","","","","nucleus","","","","","","","","","cell"])
            #X = np.zeros((21,10))
            ax[k //6, k % 6].set_title(f"{org}", fontsize=30)
        fig.suptitle(f"PC{i}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/allorg_PC{i}.png", bbox_inches="tight")
        """
        plt.tight_layout()
            fig, ax = plt.subplots(10,1)
            for sp, intensity in enumerate(intensities):
                ax[sp].imshow(intensity)          
            plt.axis("off")
            plt.tight_layout()

        
        img = plt.imread(f"{save_dir}/{org}.png")
        """
def cell_nu_ratio_cutoff():
    organelle_dir = "C:/Users/trang.le/Desktop/shapemode/organelle"
    shape_var_dir = "C:/Users/trang.le/Desktop/shapemode/U-2_OS/0"
    n_cells_per_pc = pd.read_csv(f"{organelle_dir}/cells_per_bin.csv")
    cell_nu_ratio = pd.read_csv(f"{shape_var_dir.rsplit('/',1)[0]}/cell_nu_ratio.txt", header=None)
    cell_nu_ratio.columns=["path","name","ratio"]
    plt.hist(cell_nu_ratio.ratio, bins=30, range=[0,25])
    n_coef = 128 
    inverse_func = coefs.inverse_fft 
    # Different sample rate
    d = "C:/Users/trang.le/Desktop/cellprows_mnt"
    ids = ["410_E3_4_11","410_E3_4_12","410_E3_4_13","1377_F1_3_12"]
    for im_id in ids:
        #protein_path = imread(f"{d}/{im_id}_protein.png")
        protein_path = f"{d}/{im_id}_protein.png"
        cellshape_path = f"{d}/{im_id}.npy"
        alignment.get_coefs_im(cellshape_path, save_dir, log_dir, n_coef=32, func=None, plot=False)
        nuclei_coords = 
        cell_coords = 
        fcoef_n, e_n = coefs.fourier_coeffs(nuclei_coords, n=n_coef)
        fcoef_c, e_c = coefs.fourier_coeffs(cell_coords, n=n_coef)
        
        shifts = find_line(txt_path, specific_text)
        intensity = plotting.get_protein_intensity(
            pro_path = protein_path, 
            shift_dict = shifts[l],
            ori_fft = ori_fft, 
            n_coef = n_coef, 
            inverse_func = inverse_func
            )
        
def find_line(txt_path, specific_text):
    """
    Find line containing specific text in a large txt file

    Parameters
    ----------
    txt_path : str
        path to txt file.
    specific_text : str
        text to find.

    Returns
    -------
    l : str
        first line containing specific_text.
    index : int
        index of this line in the file
    """
    with open(txt_path, "r") as fp:
        lines = fp.readlines()
        for l in lines:
            if l.find(specific_text) != -1 :
                return l, lines.index(l)
            

if __name__ == '__main__':
    main()