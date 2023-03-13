import sys
sys.path.append("..")
from coefficients import alignment,coefs
import cv2
import numpy as np
from utils import helpers
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image
from scipy.ndimage import center_of_mass, rotate
from warps import TPSpline
#from warps import TPSpline_rewrite as TPSpline

def find_landmarks(nuclei, cell, n_points=32, border_points = False):
    assert nuclei.shape == cell.shape

    # adding 1 pixel in the border for more continuous contour finding
    nu = np.zeros((nuclei.shape[0]+2, nuclei.shape[1]+2))
    nu[1:1+nuclei.shape[0],1:1+nuclei.shape[1]] = nuclei
    nu_centroid = center_of_mass(nu) 
    nu_contour = find_contours(nu, 0, fully_connected="high")
    x,y = helpers.equidistance(nu_contour[0][:,0], nu_contour[0][:,1], n_points=n_points)
    nu_contour = np.array([[x[i], y[i]] for i in range(n_points)])

    """
    convex_hull_nu = convex_hull_image(nu)
    convex_hull_nu_contour = find_contours(convex_hull_nu)
    x,y = helpers.equidistance(convex_hull_nu_contour[0][:,0], convex_hull_nu_contour[0][:,1], n_points=n_points)
    convex_hull_nu_contour = np.array([[x[i], y[i]] for i in range(len(x))])
    """
    
    cell_ = np.zeros((cell.shape[0]+2, cell.shape[1]+2))
    cell_[1:1+cell.shape[0],1:1+cell.shape[1]] = cell
    cell_contour = find_contours(cell_, 0, fully_connected="high")
    
    if len(cell_contour)>1:
        cell_contour = np.vstack(cell_contour)
        x,y = helpers.equidistance(cell_contour[:,0], cell_contour[:,1], n_points=n_points)
    else:
        x,y = helpers.equidistance(cell_contour[0][:,0], cell_contour[0][:,1], n_points=n_points)

    cell_contour = np.array([[x[i], y[i]] for i in range(n_points)])

    if border_points:
        (x_max, y_max) = cell.shape
        border_anchors = [[0,0],[x_max//2,0],[x_max,0],[0,y_max//2],[0,y_max],[x_max//2,y_max],[x_max,y_max//2],[x_max,y_max]]
        landmarks = np.vstack([np.array(nu_centroid),
                        helpers.realign_contour_startpoint(nu_contour),
                        helpers.realign_contour_startpoint(cell_contour), border_anchors])
    else:
        landmarks = np.vstack([np.array(nu_centroid),
                        helpers.realign_contour_startpoint(nu_contour),
                        helpers.realign_contour_startpoint(cell_contour)])
    return landmarks


def warp_image(pts_from, pts_to, img, midpoint=False, plot=True, save_dir=""):
    x_max = img.shape[0]
    y_max = img.shape[1]
    if midpoint:
        midpoint = (pts_from + pts_to) /2
        transform1 = TPSpline._make_inverse_warp(pts_from, midpoint, (0, 0, x_max, y_max), approximate_grid=2)
        transform2 = TPSpline._make_inverse_warp(midpoint, pts_to, (0, 0, x_max, y_max), approximate_grid=2)
        warped1 = cv2.remap(img, transform1[1].astype('float32'), transform1[0].astype('float32'), cv2.INTER_LINEAR)
        warped = cv2.remap(warped1, transform2[1].astype('float32'), transform2[0].astype('float32'), cv2.INTER_LINEAR)
        if plot:
            fig, ax = plt.subplots(1,3, figsize=(15,30))
            ax[0].imshow(img)
            ax[0].scatter(pts_from[:,1], pts_from[:,0], c=np.arange(len(pts_from)),cmap='Reds')
            ax[0].set_title('protein channel')
            ax[1].imshow(warped1)
            ax[1].scatter(midpoint[:,1], midpoint[:,0], c=np.arange(len(pts_from)),cmap='Reds')
            ax[1].set_title('ori_shape to midpoint')
            ax[2].imshow(warped)
            ax[2].scatter(pts_to[:,1], pts_to[:,0], c=np.arange(len(pts_to)),cmap='Reds')
            ax[2].set_title('midpoint to avg_shape')
    else:
        transform = TPSpline._make_inverse_warp(pts_from, pts_to, (0, 0, x_max, y_max), approximate_grid=2)
        #print("Max y: ", transform[1].astype('float32').max(), "Max x: ", transform[0].astype('float32').max(), img.max())
        warped = cv2.remap(img, transform[1].astype('float32'), transform[0].astype('float32'), cv2.INTER_LINEAR)
    return warped
