import os
from imageio import imread
import numpy as np
from utils import coefs, helpers, alignment, image_warp
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from scipy.ndimage import center_of_mass, rotate
from utils import TPSpline

def main():   
    cell_line = 'U-2 OS'
    project_dir = f"/data/2Dshapespace/{cell_line.replace(' ','_')}"
    shape_mode_path = f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/0"  
    fft_dir = f"{project_dir}/fftcoefs"  
    data_dir = f"{project_dir}/cell_masks" 
    save_dir = f"{project_dir}/U-2_OS/morphed_protein_avg" 
    n_landmarks = 32 # number of landmark points for each ring, so final n_points to compute dx, dy will be 2*n_landmarks+1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load average cell
    avg_cell = np.load(f"{shape_mode_path}/Avg_cell.npz")
    nu_centroid = [0,0]
    ix_n = avg_cell['ix_n']
    iy_n = avg_cell['iy_n']
    ix_c = avg_cell['ix_c']
    iy_c = avg_cell['iy_c']
    
    # Move average shape from zero-centered coords to min=[0,0]
    min_x = np.min(ix_c)
    min_y = np.min(iy_c)
    print(min_x,min_y)
    nu_centroid[0] -= min_x
    nu_centroid[1] -= min_y
    ix_n -= min_x
    iy_n -= min_y
    ix_c -= min_x
    iy_c -= min_y

    if len(ix_n) != n_landmarks:
        ix_n, iy_n = helpers.equidistance(ix_n, iy_n, n_points=n_landmarks)
        ix_c, iy_c = helpers.equidistance(ix_c, iy_c, n_points=n_landmarks)
    nu_contour = np.stack([ix_n, iy_n]).T
    cell_contour = np.stack([ix_c, iy_c]).T
    print(nu_contour.shape, cell_contour.shape)
    
    pts_avg = np.vstack([np.asarray(nu_centroid),
                        alignment.realign_contour_startpoint(nu_contour),
                        alignment.realign_contour_startpoint(cell_contour)])
    print(pts_avg.max(), pts_avg.min())
    
    with open(f"{fft_dir}/shift_error_meta_fft128.txt", "r") as F:
        lines = F.readlines()
    
    imlist = []
    for img_id in imlist:
        for line in enumerate(lines) :
            if line.find(img_id) != -1 :
                vals = line.strip().split(',')
                break
        theta = np.float(vals[1])
        shift_c = (np.float(vals[2].strip('(')),(np.float(vals[3].strip(')'))))

        cell_shape = np.load(f"{data_dir}/{img_id}.npy")
        img = imread(f"{data_dir}/{img_id}_protein.png")
        img = rotate(img, theta)
        print(img.shape)
        pts_ori = image_warp.find_landmarks(cell_shape[:,:,2], cell_shape[:,:,0], n_points=32, border_points = False)
        
        convex_hull_nu = convex_hull_image(cell_shape[:,:,2])
        convex_hull_cell = convex_hull_image(cell_shape[:,:,0])
        pts_convex = image_warp.find_landmarks(convex_hull_nu, convex_hull_cell, n_points=32, border_points = False)
        
        warped1 = TPSpline.warp_image(pts_ori, pts_convex, img, plot=True, save_dir="")
        warped = TPSpline.warp_image(pts_convex, pts_avg, img, plot=True, save_dir="")
        fig, ax = plt.subplots(1,4, figsize=(15,30))
        ax[0].imshow(cell_shape)
        ax[0].set_title('original segmentation')
        ax[1].imshow(img)
        ax[1].scatter(pts_ori[:,1], pts_ori[:,0], c=np.arange(len(pts_ori)),cmap='Reds')
        ax[1].set_title('protein channel')
        ax[2].imshow(warped1)
        ax[2].scatter(pts_ori[:,1], pts_ori[:,0], c=np.arange(len(pts_ori)),cmap='Reds')
        ax[2].set_title('ori_shape to midpoint')
        ax[3].imshow(warped)
        ax[3].scatter(pts_avg[:,1], pts_avg[:,0], c=np.arange(len(pts_ori)),cmap='Reds')
        ax[3].set_title('midpoint to avg_shape')
        fig.savefig(f"{save_dir}/{img_id}.png")

if __name__ == '__main__':
    main()