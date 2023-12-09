import os
import io
import json
from pickletools import uint8
import numpy as np
import math
from scipy import ndimage as ndi
import skimage
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.morphology import closing, square
from skimage.segmentation import clear_border, watershed
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# from aicsshparam import shtools
from skimage.morphology import ball, cube, octahedron

import subprocess
def grep(pattern, file_path):
    try:
        output = subprocess.check_output(['grep', pattern, file_path], universal_newlines=True)
        return output.splitlines()
    except subprocess.CalledProcessError:
        return []

def find(dirpath, prefix=None, suffix=None, recursive=True, full_path=True):
    """Function to find recursively all files with specific prefix and suffix in a directory
    Return a list of paths
    """

    l = []
    if not prefix:
        prefix = ""
    if not suffix:
        suffix = ""
    for (folders, subfolders, files) in os.walk(dirpath):
        for filename in [
            f for f in files if f.startswith(prefix) and f.endswith(suffix)
        ]:
            if full_path:
                l.append(os.path.join(folders, filename))
            else:
                l.append(filename)
        if not recursive:
            break
    return l


def read_from_json(json_file_path):
    """Function to read json file (annotation file)"""
    with io.open(json_file_path, "r", encoding="utf-8-sig") as myfile:
        data = json.load(myfile)
    return data


def watershed_lab(image, marker=None, rm_border=False):
    """Segmentation function
    Watershed algorithm to segment the 2d image based on foreground and background seed
    and use edges (sobel) as elevation map
    return labeled nuclei
    """
    # determine markers for watershed if not specified
    if marker is None:
        marker = np.full_like(image, 0)
        marker[image == 0] = 1  # background
        marker[image > threshold_otsu(image)] = 2  # foreground nuclei

    # use sobel to detect edge, then smooth with gaussian filter
    elevation_map = gaussian(sobel(image), sigma=2)

    # segmentation with watershed algorithms
    segmentation = watershed(elevation_map, marker)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_obj, num = ndi.label(segmentation)

    if rm_border is True:
        # remove bordered objects, now moved to later steps of segmentation pipeline
        bw = closing(labeled_obj > 0, square(3))
        cleared = clear_border(bw)
        labeled_obj, num = ndi.label(cleared)

    # remove too small or too large object
    output = np.zeros(labeled_obj.shape)
    for region in regionprops(labeled_obj):
        if region.area >= 2000:  # <= thres_high:
            # if the component has a volume within the two thresholds,
            # set the output image to 1 for every pixel of the component
            output[labeled_obj == region.label] = 1

    labeled_obj, num = ndi.label(output)

    return labeled_obj, num


def watershed_lab2(image, marker=None):
    """Watershed segmentation with topological distance
    and keep the relative ratio of the cells to each other
    return labeled cell body, each has 1 nuclei
    """
    distance = ndi.distance_transform_edt(image)  # if the cells are sparse
    # distance = ndi.distance_transform_edt(marker) #if the cells are crowded
    distance = clear_border(distance, buffer_size=50)
    # determine markers for watershed if not specified
    if marker is None:
        local_maxi = peak_local_max(
            distance, indices=False, footprint=np.ones((3, 3)), labels=image
        )
        marker = ndi.label(local_maxi)[0]

    # segmentation with watershed algorithms
    segmentation = watershed(-distance, marker, mask=image)

    return segmentation


def find_border(labels, buffer_size=0, bgval=0, in_place=False):
    """Find indices of objects connected to the label image border.
    Adjusted from skimage.segmentation.clear_border()
    Parameters
    ----------
    labels : (M[, N[, ..., P]]) array of int or bool
        Imaging data labels.>
    buffer_size : int, optional
        The width of the border examined.  By default, only objects
        that touch the outside of the image are removed.
    bgval : float or int, optional
        Cleared objects are set to this value.
    in_place : bool, optional
        Whether or not to manipulate the labels array in-place.
    Returns
    -------
    out : (M[, N[, ..., P]]) array
        Imaging data labels with cleared borders
    """
    image = labels

    if any((buffer_size >= s for s in image.shape)):
        raise ValueError("buffer size may not be greater than image size")

    # create borders with buffer_size
    borders = np.zeros_like(image, dtype=np.bool_)
    ext = buffer_size + 1
    slstart = slice(ext)
    slend = slice(-ext, None)
    slices = [slice(s) for s in image.shape]
    for d in range(image.ndim):
        slicedim = list(slices)
        slicedim[d] = slstart
        borders[slicedim] = True
        slicedim[d] = slend
        borders[slicedim] = True

    # Re-label, in case we are dealing with a binary image
    # and to get consistent labeling
    # labels = skimage.measure.label(image, background=0)

    # determine all objects that are connected to borders
    borders_indices = np.unique(labels[borders])

    return borders_indices


def image_roll(img_1_ch, cm):
    Shift = np.zeros_like(img_1_ch)
    c = [img_1_ch.shape[0] / 2.0, img_1_ch.shape[1] / 2.0]
    S = np.roll(img_1_ch, int(round(c[0] - cm[0])), axis=0)
    S = np.roll(S, int(round(c[1] - cm[1])), axis=1)
    Shift = S
    return Shift


def shift_center_mass(image):
    """Function to center images to the Nuclei center of mass
    assuming channel 2 is the nuclei channel
    If there is only 1 channel then center to the center of mass of the blob
    """
    if image.ndim == 2:
        cm = ndi.measurements.center_of_mass(image)
        Shift = image_roll(image, cm)
    elif image.ndim == 3:
        # img = np.asanyarray(image)
        cm_nu = ndi.measurements.center_of_mass(image[:, :, 2])
        Shift = np.zeros_like(image)
        for channel in (0, 1, 2):
            im = image[:, :, channel]
            Shift[:, :, channel] = image_roll(im, cm_nu)

    return Shift


def interpolate_(data, centroid_n):
    cell = Polygon()
    isIntersection = poly1.intersection(Line(p1, Point(3, 2)))
    return image


def find_min(coords, centroid):
    dis = []
    for k in range(len(coords)):
        dis += math.hypot(coords[k][0] - centroid[0], coords[k][1] - centroid[1])
    p_shortest = [np.argmin(dis)]
    # angle =
    return p_shortest


def resize_pad(image, ratio=0.25, size=256):
    """Function to resize and pad segmented image, keeping the aspect ratio
    and keep the relative ratio of the cells to each other
    """
    """
    #input an Image object (PIL)
    image = Image.fromarray(image)
    desired_size = size 
    
    # current size of the image
    old_size=image.size
    # old_size[0] is in (width, height) format
    ratio = 0.25 #float(desired_size)/max(old_size) 
    
    # size of the image after reduced by ratio
    new_size = tuple([int(x*ratio) for x in old_size])
    
    # resize image
    im = image.resize(new_size, Image.ANTIALIAS)
    
    # create a new blank image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    """
    im = skimage.transform.rescale(image, ratio)
    if im.ndim == 2:
        new_im = np.zeros((size, size))
        dif_axis0 = (size - im.shape[0]) // 2
        dif_axis1 = (size - im.shape[1]) // 2
        new_im[
            dif_axis0 : dif_axis0 + im.shape[0], dif_axis1 : dif_axis1 + im.shape[1]
        ] = im

    elif im.ndim == 3:
        new_im = np.zeros((size, size, 3))

        dif_axis0 = (size - im.shape[0]) // 2
        dif_axis1 = (size - im.shape[1]) // 2

        if (size > im.shape[0]) & (size > im.shape[1]):
            # if segmented cell is smaller than desired size:
            new_im[
                dif_axis0 : dif_axis0 + im.shape[0],
                dif_axis1 : dif_axis1 + im.shape[1],
                :,
            ] = im
        elif (size < im.shape[0]) & (size > im.shape[1]):
            im = im[np.abs(dif_axis0) : np.abs(dif_axis0) + size, :, :]
            new_im[:, dif_axis1 : dif_axis1 + im.shape[1], :] = im
        elif (size > im.shape[0]) & (size < im.shape[1]):
            im = im[:, np.abs(dif_axis1) : np.abs(dif_axis1) + size, :]
            new_im[dif_axis0 : dif_axis0 + im.shape[0], :, :] = im
        elif (size < im.shape[0]) & (size < im.shape[1]):
            new_im = im[
                np.abs(dif_axis0) : np.abs(dif_axis0) + size,
                np.abs(dif_axis1) : np.abs(dif_axis1) + size,
                :,
            ]

    return new_im  # this is  float64


def curvelet_transform(
    x, num_bands, num_angles=8, all_curvelets=True, as_complex=False
):
    """
    From https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/curvelet.py
    """
    ndims = len(x.shape)

    # This file requires Curvelab and the PyCurveLab packages be installed on your system.
    try:
        import pyct
    except ImportError:
        raise NotImplementedError(
            "Use of curvelets requires installation of CurveLab and the PyCurveLab package.\nSee: http://curvelet.org/  and  https://www.slim.eos.ubc.ca/SoftwareLicensed/"
        )

    if ndims == 2:
        ct = pyct.fdct2(
            n=x.shape,
            nbs=num_bands,  # Number of bands
            nba=num_angles,  # Number of discrete angles
            ac=all_curvelets,  # Return curvelets at the finest detail level
            vec=False,  # Return results as nested python vectors
            cpx=as_complex,
        )  # Disable complex-valued curvelets
    elif ndims == 3:
        ct = pyct.fdct3(
            n=x.shape,
            nbs=num_bands,  # Number of bands
            nba=num_angles,  # Number of discrete angles
            ac=all_curvelets,  # Return curvelets at the finest detail level
            vec=False,  # Return results as nested python vectors
            cpx=as_complex,
        )  # Disable complex-valued curvelets
    else:
        raise NotImplementedError("%dD Curvelets are not supported." % (ndims))
    result = ct.fwd(x)
    del ct
    return result


def flip_signs(A, B):
    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs


def normalize_complex_arr(a):
    # Normalize complex array from 0+0j to 1+1*J
    a_oo = a - a.real.min() - 1j * a.imag.min()  # origin offsetted
    return a_oo / np.abs(a_oo).max()


def find_centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (_x, _y)

def euclidean_distance(point1, point2):
    """ point1, point2: (x, y)
    """
    return math.sqrt((point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2)

def equidistance(x, y, n_points=256):
    distance = np.cumsum(
        np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2)
    )

    distance = distance / distance[-1]

    fx = interp1d(distance, x)
    fy = interp1d(distance, y)

    alpha = np.linspace(0, 1, n_points)
    x_regular, y_regular = fx(alpha), fy(alpha)
    return x_regular, y_regular


def P2R(radii, angles):
    """Function to handle complex numbers
    turn magnitude and angle to complex number
    """
    return radii * np.exp(1j * angles)


def R2P(x):
    """Function to handle complex numbers
    turn complex number x to magnitude and angle
    """
    return abs(x), np.angle(x)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)


def find_nearest(array, value):
    """Function to find nearest index and item in an array so that the item is clost to value
    Return idx, item
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_random_3d_shape():
    idx = np.random.choice([0, 1, 2], 1)[0]
    element = [ball, cube, octahedron][idx]
    label = ["ball", "cube", "octahedron"][idx]
    img = element(10 + int(10 * np.random.rand()))
    img = np.pad(img, ((1, 1), (1, 1), (1, 1)), mode="constant")
    img = img.reshape(1, *img.shape)
    # Rotate shapes to increase dataset variability.
    img = shtools.rotate_image_2d(image=img, angle=360 * np.random.rand()).squeeze()
    return label, img


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


def plot_3d(array_3d):
    ax = make_ax(True)
    ax.voxels(array_3d, edgecolors="gray")
    plt.show()


def get_location_counts(locations_list, all_locations):

    label_counts = dict.fromkeys(all_locations, 0)
    for locations in locations_list:
        if locations != "Discard":
            sc_locations = []
            idx_list = locations.split("|")
            for idx in idx_list:
                sc_locations += [
                    k for k, v in all_locations.items() if v == int(float(idx))
                ]

            for l in sc_locations:
                label_counts[l] += 1
    return label_counts


def rgb_2_gray_unique(image, channel_last=True):
    """ Function to convert RGB into gray image, but each unique rgb pixel is a unique pixel number in gray image

    Args:
        image: RGB image (width,height,3)
        channel_last: position of the channel axis
    Returns: (width,height)
    """
    if not channel_last:
        image = np.moveaxis(image, 0, -1)

    flat_img = image.reshape(-1, 3)
    unique_px = np.unique(flat_img, axis=0)  # [0,0,0] always the first unique px value
    gray = np.zeros(
        (image.shape[0] * image.shape[1]), dtype=np.uint8
    )  # (width*height,)
    for i, px_val in enumerate(unique_px):
        indexes = np.where((flat_img == px_val).sum(axis=-1) == 3)[0]
        gray[indexes] = i
    gray = gray.reshape((image.shape[0], image.shape[1]))
    assert len(unique_px) == len(np.unique(gray))
    # print(f"before: {len(unique_px)} unique px, after: {len(np.unique(gray))} unique values")
    # print(gray.shape, gray.dtype)
    return gray


def flatten_list(list_of_lists):
    """ 
    Function to flatten list of lists, element can be 1 item of a list of items

    Args:
        list_of_lists: list of lists
    Returns: list containing all sub-items
    """
    l = [item for sublist in list_of_lists for item in sublist]
    return l


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def realign_contour_startpoint(xy, nearest_p=None):
    x = np.array([p[0] for p in xy])
    y = np.array([p[1] for p in xy])
    if nearest_p is None:
        centroid = find_centroid(xy)
        _, val = find_nearest(y[np.where(x > centroid[0])], centroid[1])
    else:        
        _, val = find_nearest(y[np.where(x > nearest_p[0])], nearest_p[1])
    if len(np.where(y == val)[0]) > 1:
        largest_x = x.min()
        current_idx = None
        for idx in np.where(y == val)[0]:
            if x[idx] > largest_x:
                largest_x = x[idx]
                current_idx = idx
        idx = current_idx
    else:
        idx = np.where(y == val)[0][0]

    xy = np.concatenate((xy, xy))[idx : idx + len(xy)]
    return xy


def get_line(file_path, search_text="", mode="first"):
    # Function to read line(s) containing search_text in a very large txt file (x GB)
    with open(file_path, "r") as F:
        lines = F.readlines()
    if mode == "first":
        for line in lines:
            if line.find(search_text) != -1:
                return line
    elif mode == "all":
        l_results = []
        for line in lines:
            if line.find(search_text) != -1:
                l_results += [line]
        return l_results

def factoring_data_into_tertiles(data_points):
    n = len(data_points)
    # Sort the data points in ascending order
    sorted_data_points = sorted(data_points)
    
    # Divide the sorted data points into tertiles
    tertile_1 = sorted_data_points[:n//3]
    tertile_2 = sorted_data_points[n//3:2*n//3]
    tertile_3 = sorted_data_points[2*n//3:]
    
    # Create a list of indexes for the first 3 tertiles
    indexes_1 = []
    indexes_2 = []
    indexes_3 = []
    for i, point in enumerate(data_points):
        if point in tertile_1:
            indexes_1.append(i)
        elif point in tertile_2:
            indexes_2.append(i)
        elif point in tertile_3:
            indexes_3.append(i)
    
    return indexes_1, indexes_2, indexes_3

def get_items_from_list_by_index(input_list, indexes):
    return list(map(lambda x:input_list[x],indexes))