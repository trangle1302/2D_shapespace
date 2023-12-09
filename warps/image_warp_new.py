import sys
sys.path.append("..")
#import cv2
import numpy as np
from utils import helpers
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image
from scipy.ndimage import center_of_mass, map_coordinates

# from warps import TPSpline
# from warps import TPSpline_rewrite as TPSpline
from warps import tps


def find_landmarks(nuclei, cell, n_points=32, border_points=False):
    assert nuclei.shape == cell.shape

    # adding 1 pixel in the border for more continuous contour finding
    nu = np.zeros((nuclei.shape[0] + 2, nuclei.shape[1] + 2))
    nu[1 : 1 + nuclei.shape[0], 1 : 1 + nuclei.shape[1]] = nuclei
    nu_centroid = center_of_mass(nu)
    nu_contour = find_contours(nu) #, 0, fully_connected="high")
    x, y = helpers.equidistance(
        nu_contour[0][:, 0], nu_contour[0][:, 1], n_points=n_points + 1
    )
    nu_contour = np.array([[x[i], y[i]] for i in range(n_points)]) # first and last point is the same

    cell_ = np.zeros((cell.shape[0] + 2, cell.shape[1] + 2))
    cell_[1 : 1 + cell.shape[0], 1 : 1 + cell.shape[1]] = cell
    cell_contour = find_contours(cell_)#, 0, fully_connected="high")

    if len(cell_contour) > 1:
        print(f'broken contours to {len(cell_contour)} fragments, {[len(x) for x in cell_contour]}')
        #cell_contour = np.vstack(cell_contour)
        cell_contour = cell_contour[np.argmax([len(x) for x in cell_contour])]
        print(f'Picked the largest contour line {len(cell_contour)}')
        x, y = helpers.equidistance(
            cell_contour[:, 0], cell_contour[:, 1], n_points=n_points * 2 + 1
        )
    else:
        x, y = helpers.equidistance(
            cell_contour[0][:, 0], cell_contour[0][:, 1], n_points=n_points * 2 + 1
        )
    cell_contour = np.array([[x[i], y[i]] for i in range(n_points * 2)]) # first and last point is the same
    # print('Fist+last points:', x[0], x[-1], y[0], y[-1])
    if border_points:
        (x_max, y_max) = cell.shape
        border_anchors = [
            [0, 0],
            #[x_max // 2, 0],
            [x_max, 0],
            #[0, y_max // 2],
            [0, y_max],
            #[x_max // 2, y_max],
            #[x_max, y_max // 2],
            [x_max, y_max],
        ]
        landmarks = np.vstack(
            [
                np.array(nu_centroid),np.array(nu_centroid),
                helpers.realign_contour_startpoint(nu_contour),
                helpers.realign_contour_startpoint(cell_contour, nearest_p=None),
                border_anchors,
            ]
        )
    else:
        landmarks = np.vstack(
            [
                np.array(nu_centroid),
                helpers.realign_contour_startpoint(nu_contour),
                helpers.realign_contour_startpoint(cell_contour, nearest_p=None),
            ]
        )
    return landmarks


def warp_image(pts_from, pts_to, img):
    tps_f = tps.ThinPlateSpline(alpha=0.1)  # 0 Regularization
    tps_f.fit(pts_to, pts_from)
    x, y = img.shape
    # x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]
    t_grid = np.indices((x, y), dtype=np.float64).transpose(1, 2, 0)
    from_grid = tps_f.transform(t_grid.reshape(-1, 2)).reshape(x, y, 2)
    warped = map_coordinates(img, from_grid.transpose(2, 0, 1))
    return warped
