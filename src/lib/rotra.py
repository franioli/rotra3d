import numpy as np

from lmfit import Parameters

from lib.utils import convert_to_homogeneous
from lib.transformations import (
    euler_matrix,
    euler_from_matrix,
    affine_matrix_from_points,
)


def compute_approx_values(
    pt_loc: np.ndarray,
    pt_world: np.ndarray,
):

    T = affine_matrix_from_points(pt_loc, pt_world,
                                  shear=False, scale=True, usesvd=True)
    t = T[:3, 3:4].squeeze()
    rot = euler_from_matrix(T[:3, :3])
    m = float(1.0)

    return (rot, t, m)


def get_T_from_params(
    params: Parameters,
):

    # Get parameters
    parvals = params.valuesdict()
    rx = parvals['rx']
    ry = parvals['ry']
    rz = parvals['rz']
    tx = parvals['tx']
    ty = parvals['ty']
    tz = parvals['tz']
    m = parvals['m']

    # Build 4x4 transformation matrix (T) in homogeneous coordinates
    T = np.identity(4)
    R = euler_matrix(rx, ry, rz)
    T[0:3, 0:3] = (m * np.identity(3)) @ R[:3, :3]
    T[0:3, 3:4] = np.array([tx, ty, tz]).reshape(3, 1)

    return T


def compute_residuals(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    weights: np.ndarray = None,
    prior_covariance_scale: float = None,
) -> np.ndarray:
    ''' 3D rototranslation with scale factor

    X1_ = T_ + m * R * X0_

    Inputs: 
    - x0 (np.ndarray): Points in the starting reference system
    - x1 (np.ndarray): Points in final reference system 
    - weights (np.ndarray, defult = None): weights (e.g., inverse of a-priori observation uncertainty)
    - prior_covariance_scale (float, default = None): A-priori sigma_0^2     

    Return: 
    - res (nx1 np.ndarray): Vector of the weighted residuals

    '''

    # Get parameters
    parvals = params.valuesdict()
    rx = parvals['rx']
    ry = parvals['ry']
    rz = parvals['rz']
    tx = parvals['tx']
    ty = parvals['ty']
    tz = parvals['tz']
    m = parvals['m']

    # Convert points to homogeneos coordinates and traspose np array to obtain a 4xn matrix
    x0 = convert_to_homogeneous(x0).T

    # Build 4x4 transformation matrix (T) in homogeneous coordinates
    T = np.identity(4)
    R = euler_matrix(rx, ry, rz)
    T[0:3, 0:3] = (m * np.identity(3)) @ R[:3, :3]
    T[0:3, 3:4] = np.array([tx, ty, tz]).reshape(3, 1)

    # Apply transformation to x0 points
    x1_ = T @ x0
    x1_ = x1_[:3, :].T

    # Compute residuals as differences between observed and estimated values, scaled by the a-priori observation uncertainties
    res = (x1 - x1_)

    # If weigthts are provided, scale residual
    if weights is not None:

        if weights.shape != res.shape:
            raise ValueError(
                f'Wrong dimensions of the weight matrix. It must be of size {res.shape}')

        res = res * weights

    return res.flatten()
