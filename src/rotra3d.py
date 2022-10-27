import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from lmfit import Minimizer, Parameters

from lib.rotra import (
    compute_residuals,
    compute_approx_values,
    get_T_from_params,
)
from lib.io import read_data_to_df
from lib.utils import print_results


def main(
    pt_loc: pd.DataFrame,
    pt_world: pd.DataFrame,
    uncertainty: pd.DataFrame,
):

    rot_ini, t_ini, m_ini = compute_approx_values(
        pt_loc.to_numpy().T,
        pt_world.to_numpy().T
    )
    print('Initial values:')
    print('Rotations:\n', rot_ini)
    print('Translation:\n', t_ini)

    # Define Parameters to be optimized
    params = Parameters()
    params.add('rx', value=rot_ini[0], vary=True)
    params.add('ry', value=rot_ini[1], vary=True)
    params.add('rz', value=rot_ini[2], vary=True)
    params.add('tx', value=t_ini[0], vary=True)
    params.add('ty', value=t_ini[1], vary=True)
    params.add('tz', value=t_ini[2], vary=True)
    params.add('m',  value=m_ini, vary=True)

    # A-priori Sigma_0²: scale of the covariance matrix
    sigma0_2 = 1

    # Run Optimization!
    weights = 1. / uncertainty.to_numpy()
    minimizer = Minimizer(
        compute_residuals,
        params,
        fcn_args=(
            pt_loc.to_numpy(),
            pt_world.to_numpy(),
        ),
        fcn_kws={
            'weights': weights,
            'prior_covariance_scale': sigma0_2,
        },
        scale_covar=True,
    )
    result = minimizer.minimize(method='leastsq')
    # fit_report(result)

    # Print result
    print_results(result, weights, sigma0_2)

    return get_T_from_params(result.params)


if __name__ == '__main__':

    pt_loc = read_data_to_df(
        'data/belpy/lingua_loc_old.txt',  # 'data/belpy/loc.txt',
        delimiter=',',
        header=None,
        col_names=['x', 'y', 'z'],
        index_col=0,
    )
    pt_world = read_data_to_df(
        'data/belpy/lingua_utm_old.txt',
        delimiter=',',
        header=None,
        col_names=['x', 'y', 'z'],
        index_col=0
    )
    print('Point world:\n', pt_world)
    print('Point loc:\n', pt_loc)

    # Define Weights as the inverse of the a-priori standard deviation of each observation
    # All the measurements are assumed as independent (Q diagonal)
    # Weight matrix must have the same shape as the observation array X0
    # Default assigned uncertainty[m]
    uncertainty = 0.02 * np.ones(pt_loc.shape)
    uncertainty = pd.DataFrame(uncertainty)
    uncertainty.columns = ['sx', 'sy', 'sz']
    uncertainty.index = pt_loc.index
    # uncertainty.loc['TAB2'] = [0.05, 0.05, 0.05]
    # uncertainty.loc['TAB3'] = [0.05, 0.05, 0.05]
    print(f'Prior uncertainties:\n{uncertainty}')

    # # Remove observations
    # rows_to_drop = ['F1', 'F1BIS']
    # pt_loc = pt_loc.drop(labels=rows_to_drop)
    # pt_world = pt_world.drop(labels=rows_to_drop)
    # uncertainty = uncertainty.drop(labels=rows_to_drop)
    # print(f'Points loc: \n {pt_loc}')

    print('\n--------------\n')
    print('Starting computation:')

    T = main(pt_world, pt_loc, uncertainty)

    print('Trasnsformation matrix:\n', T)

    # pts = np.array([
    #     [15977.431],
    #     [90863.486],
    #     [2007.726],
    #     [1., ],
    # ])
    pts = np.array([
        [16296.2032, 90660.0410000002, 2002.208, 1.],
        [15977.4310, 90863.4859999996, 2007.726, 1.],
        [16087.7726, 91031.6390000004, 1989.334, 1.],
        [15972.4999, 90806.7769999998, 1985.165, 1.],
    ]).T

    pts_est = T @ pts
    print(pts_est.T)
