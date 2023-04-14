"""
Implementation of a thin-plate-spline warping transform that warps from the from_points
    to the to_points, and then warp the given images by that transform. 
    Original paper: "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations" by F.L. Bookstein.
    Useful blogpost: https://profs.etsmtl.ca/hlombaert/thinplates/
"""
import cv2
import numpy as np
from scipy.dis

def _U(r): 
    """ Radial basis function (smooth kernels centered around control points)
    r^2 log(r)
    """
    #print(r.min(), r.max())
    #return np.power(r,2) * np.log(r)
    return (r**2) * np.where(r<1e-100, 0, np.log(r))

def _K(points):
    """ Surface function
    Parameters:
        points: coordinates (x,y) in any convenient cartesian coordinate system
    Returns:
        K: N x N
    """
    # r_ is the matrix with distance between point i and j
    # r_ij = |Pi - Pj|
    dx = np.subtract.outer(points[:,0],points[:,1])
    dy = np.subtract.outer(points[:,1],points[:,1])
    r_ = np.sqrt(dx**2 + dy**2)
    return _U(r_)

def _L(points):
    N = len(points)
    K = _K(points)
    P = np.concatenate((np.ones((N,1)), points), axis=1)
    O = np.zeros((3,3))
    L = np.bmat([[K,P],[P.T, O]])
    return L

def _f_xy(points, xx, yy, Wa):
    """ Each smooth function f is divided into 2 parts: 
    sum of function U(r) (bound) and affine part representing f-> infinity 
    Parameters:
        points: from_points coordinates (x,y) in any convenient cartesian coordinate system
        xx: to_point x coordinate mesh
        yy: to_point y coordinate mesh
        Wa: [W | a1 ax ay]^T 

    Returns:
        predicted value mesh
    """
    W = Wa[:-3]
    a1 = Wa.item((0,0))
    ax = Wa.item((1,0))
    ay = Wa.item((2,0))
    sum_ = np.zeros_like(xx)
    for i, Pi in enumerate(points):
        wi = W[i,0]
        sum_ += wi * _U(np.sqrt((xx-Pi[0])**2 + (yy-Pi[1])**2))
    f_xy = a1 + ax*xx + ay*yy + sum_
    return f_xy

def warp_f(from_points, to_points, mesh_x, mesh_y):
    """
        L = K * [W | a1 ax ay]^T = [V | 0 0 0]^T
    """
    # Calculate matrix L
    L = _L(from_points)
    # V is the destination landmark, Nx2
    # Y = [V | 0 0 0]^T
    Y = np.concatenate((to_points, np.zeros((3,2))), axis=0)
    # weights [W | a1 ax ay]
    Wa = np.dot(np.linalg.pinv(L), Y)
    #print(from_points.shape, to_points.shape, Y.shape)    
    #print("WA shape: ",Wa.shape)
    x_warp = _f_xy(from_points, mesh_x, mesh_y, Wa[:,0])
    y_warp = _f_xy(from_points, mesh_x, mesh_y, Wa[:,1])
    return x_warp, y_warp

def inverse_warp(from_points, to_points, output_bbox, approximate_grid):
    assert from_points.shape == to_points.shape
    x_min, y_min, x_max, y_max = output_bbox

    # higher approximate_grid means coarser deformation fields (x_warp, y_warp)
    if approximate_grid is None: approximate_grid = 1
    x_steps = (x_max - x_min) // approximate_grid
    y_steps = (y_max - y_min) // approximate_grid
    x, y = np.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j]
    
    # calcuate deformation fields (x_warp and y_warp) from the to_points to the from_points
    # (later remap with cv2.remap)
    x_warp, y_warp = warp_f(to_points, from_points, x, y)
    #print(x_warp, y_warp)
    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        transform_x, transform_y = interpolate_deformation_grids(x_warp, y_warp, (x_max-x_min, y_max-y_min), method="linear")
        transform = [transform_x, transform_y]
    return transform 

def interpolate_deformation_grids(x_grid, y_grid, new_size, method="linear"):
    from scipy.interpolate import griddata
    # Get the original size of the deformation grids
    orig_size = x_grid.shape

    # Create new x and y coordinate grids based on the new size
    new_x = np.linspace(0, orig_size[1] - 1, new_size[1])
    new_y = np.linspace(0, orig_size[0] - 1, new_size[0])
    new_x_grid, new_y_grid = np.meshgrid(new_x, new_y)

    # Reshape the original deformation grids to 1D arrays
    orig_x_grid = x_grid.reshape(-1)
    orig_y_grid = y_grid.reshape(-1)

    # Interpolate the deformation grids at the new coordinates
    new_x_deform = griddata((np.arange(orig_size[1]), np.arange(orig_size[0])),
                            orig_x_grid, (new_x_grid, new_y_grid), method=method)
    new_y_deform = griddata((np.arange(orig_size[1]), np.arange(orig_size[0])),
                            orig_y_grid, (new_x_grid, new_y_grid), method=method)

    # Reshape the interpolated grids to the desired size
    new_x_deform = new_x_deform.reshape(new_size)
    new_y_deform = new_y_deform.reshape(new_size)

    return new_x_deform, new_y_deform

def warp_image(from_pts, to_pts, image, output_bbox, interpolation_order = 1, approximate_grid=2):
    """
    Parameters:
        - from_pts and to_pts: Nx2 arrays containing N 2D landmark points.
        - image: image to wrap with the given warp transform
        - output_bbox: the (xmin, ymin, xmax, ymax) region of the output
                image that should be produced. (Note: The region is inclusive, i.e.
                xmin <= x <= xmax)
        - interpolation_order: according to cv:
            cv::INTER_NEAREST = 0,
            cv::INTER_LINEAR = 1,
            cv::INTER_CUBIC = 2,
            cv::INTER_AREA = 3,
            cv::INTER_LANCZOS4 = 4,
            cv::INTER_LINEAR_EXACT = 5,
            cv::INTER_NEAREST_EXACT = 6,
            cv::INTER_MAX = 7,
            cv::WARP_FILL_OUTLIERS = 8,
            cv::WARP_INVERSE_MAP = 16
        - approximate_grid: what dividing order is the warp calculated. 
            Eg. approximate_grid=1 -> calculate warp on full size.
                approximate_grid=2 -> calculate warp on half of the size, then (bilinearly) interpolate to to the original region
            small approximate_grid means more computation per warp
    
    Returns:
        - warped image with the same size as output_shape
    """
    transform_x, transform_y = inverse_warp(from_pts, to_pts, output_bbox, approximate_grid)
    if interpolation_order == 0:
        intp = cv2.INTER_NEAREST
    elif interpolation_order == 1:
        intp = cv2.INTER_LINEAR
    elif interpolation_order == 2:
        intp = cv2.INTER_CUBIC
    else:
        raise NotImplementedError
    return cv2.remap(image, transform_x.astype('float32'), transform_y.astype('float32'), intp)
