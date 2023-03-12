import cv2
import numpy as np

def _U(r): 
    """ Radial basis function (smooth kernels centered around control points)
    r^2 log(r)
    """
    #print(r.min(), r.max())
    return np.power(r,2) * np.log(r)

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
        new_x, new_y = np.mgrid[x_min:x_max+1, y_min:y_max+1]
        x_fracs, x_indices = np.modf((x_steps-1)*(new_x-x_min)/float(x_max-x_min))
        y_fracs, y_indices = np.modf((y_steps-1)*(new_y-y_min)/float(y_max-y_min))
        x_indices = x_indices.astype(int)
        y_indices = y_indices.astype(int)
        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        ix1 = (x_indices+1).clip(0, x_steps-1)
        iy1 = (y_indices+1).clip(0, y_steps-1)
        t00 = x_warp[(x_indices, y_indices)]
        t01 = x_warp[(x_indices, iy1)]
        t10 = x_warp[(ix1, y_indices)]
        t11 = x_warp[(ix1, iy1)]
        transform_x = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs
        #print(transform_x)
        t00 = y_warp[(x_indices, y_indices)]
        t01 = y_warp[(x_indices, iy1)]
        t10 = y_warp[(ix1, y_indices)]
        t11 = y_warp[(ix1, iy1)]
        transform_y = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs

        transform = [transform_x, transform_y]
    return transform 

def warp_image(from_pts, to_pts, image, output_bbox, interpolation_order = 1, approximate_grid=2):
    """Implementation of a thin-plate-spline warping transform that warps from the from_points
    to the to_points, and then warp the given images by that transform. 
    The paper: "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations" by F.L. Bookstein.

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



'''
def tps_transform(X, Y, lamda=0.01):
    """
    This function calculates the Thin Plate Spline transformation for given input and target points.
    Parameters:
        X: array-like, shape=(n_samples, n_features), input points.
        Y: array-like, shape=(n_samples, n_features), target points.
        lamda: float, regularization parameter to avoid overfitting, default=0.01.
    Returns:
        W: array-like, shape=(n_samples,), weights to apply the transformation.
        A: array-like, shape=(n_samples, n_samples), matrix representing the transformation.
    """
    # calculate pairwise distances between input points
    pairwise_dist = cdist(X, X)
    # calculate K matrix
    K = pairwise_dist ** 2 * np.log(pairwise_dist + 1e-6)
    # add regularization to K matrix
    K += np.eye(X.shape[0]) * lamda
    # add ones column to X
    P = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # calculate L matrix
    L = np.concatenate((np.concatenate((K, P), axis=1), np.concatenate((P.T, np.zeros((3, 3))), axis=1)), axis=0)
    # calculate Y matrix
    Y_aug = np.concatenate((Y, np.zeros((3, Y.shape[1]))), axis=0)
    # calculate weights
    W = solve(L, Y_aug)
    # calculate transformation matrix
    A = W[:X.shape[0], :]
    return W, A

def tps_apply(X, X_new, W, A):
    """
    This function applies the Thin Plate Spline transformation to new points.
    Parameters:
        X: array-like, shape=(n_samples, n_features), input points.
        X_new: array-like, shape=(n_samples, n_features), new points to transform.
        W: array-like, shape=(n_samples,), weights to apply the transformation.
        A: array-like, shape=(n_samples, n_samples), matrix representing the transformation.
    Returns:
        Y_new: array-like, shape=(n_samples, n_features), transformed new points.
    """
    # calculate pairwise distances between new points and input points
    pairwise_dist = cdist(X_new, X)
    # calculate K matrix
    K = pairwise_dist ** 2 * np.log(pairwise_dist + 1e-6)
    # add ones column to X_new
    P_new = np.concatenate((np.ones((X_new.shape[0], 1)), X_new), axis=1)
    # calculate Y_new
    Y_new = np.dot(K, W[:X.shape[0], :]) + np.dot(P_new, A)
    return Y_new

def calculate_deformation_fields(X1, X2, n_landmarks):
    """
    This function calculates the deformation fields of X and Y in 2 images based on n landmark points.
    Parameters:
        X1: array-like, shape=(n_samples, n_features), the n landmark points in the first image.
        X2: array-like, shape=(n_samples, n_features), the n landmark points in the second image.
        n_landmarks: int, the number of landmark points.
    Returns:
        dx: array-like, shape=(n_landmarks,), deformation field of X.
        dy: array-like, shape=(n_landmarks,), deformation field of Y.
    """
    # calculate the transformation matrix A
    _, A = tps_transform(X1, X2)
    # apply the transformation matrix A to the landmark points in X1
    X1_transformed = np.dot(A, np.concatenate((np.ones((n_landmarks, 1)), X1), axis=1).T).T
    # calculate the deformation fields by subtracting the transformed landmark points from the original landmark points
    dx = X1_transformed[:, 1] - X2[:, 0]
    dy = X1_transformed[:, 2] - X2[:, 1]
    return dx, dy

'''