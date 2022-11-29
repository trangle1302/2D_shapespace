import cv2
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
    shape_mode_path = f"{project_dir}/shapemode/{cell_line.replace(' ','_')}/ratio8"   
    data_dir = f"{project_dir}/cell_masks" 
    data_dir = f"{project_dir}/cell_masks" 
    imlist = []
    average_shape = 

    nuclei = 0
    cell = 

    pts_avg = image_warp.find_landmarks(nuclei, cell, n_points=32, border_points = False)
    
    for img_id in imlist:
        cell_shape = np.load(f"{data_dir}/{img_id}.npy")
        img = imread(f"{data_dir}/{img_id}_protein.png")
        
        pts_ori = image_warp.find_landmarks(cell_shape[:,:,2], cell_shape[:,:,0], n_points=32, border_points = False)
        
        convex_hull_nu = convex_hull_image(cell_shape[:,:,2])
        convex_hull_cell = convex_hull_image(cell_shape[:,:,0])
        pts_convex = image_warp.find_landmarks(convex_hull_nu, convex_hull_cell, n_points=32, border_points = False)
        
        warped1 = TPSpline.warp_image(pts_ori, pts_convex, img, plot=True, save_dir="")
        warped = TPSpline.warp_image(pts_convex, pts_avg, img, plot=True, save_dir="")
        fig, ax = plt.subplots(1,5, figsize=(15,30))
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
        ax[4].imshow(average_shape)
        ax[4].set_title('average shape')
        fig.savefig(f"{save_dir}/tmp")

if __name__ == '__main__':
    main()