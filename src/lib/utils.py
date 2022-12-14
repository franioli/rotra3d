import numpy as np
from scipy import stats

from lmfit import fit_report


def convert_to_homogeneous(x: np.ndarray) -> np.ndarray:
    ''' Convert 3d points in euclidean coordinates (nx3 numpy array) homogenous coordinates (nx4 numpy array)
    '''
    if x.shape[1] != 3:
        raise ValueError(
            'Wrong dimension of the input array, please provide nx3 numpy array')
    n = x.shape[0]
    x = np.concatenate((x, np.ones((n, 1))), 1, )

    return x


def rescale_residuals(residuals: np.ndarray,
                      weights: np.ndarray,
                      ) -> np.ndarray:

    residuals = residuals / weights
    return residuals


def print_results(
    result,
    weights: np.ndarray = None,
    sigma0_2: float = 1.,
) -> None:

    ndim = weights.shape[1]

    # Rescale residuals
    if weights is not None:
        residuals = rescale_residuals(
            result.residual.reshape(-1, ndim),
            weights,
        )
    else:
        residuals = result.residual.reshape(-1, ndim)

    print('-------------------------------')
    print('Optimization report')
    print(fit_report(result))

    print('-------------------------------')
    print(f'Chi quadro test:')
    nfree = result.nfree
    chi_lim = stats.chi2.ppf(1-.05, df=nfree)
    chi_0 = result.redchi / sigma0_2
    print(f'Degrees of freedom: {nfree}')
    print(f'Chi2 empirical: {chi_0:.3f}')
    print(f'Chi2 limit: {chi_lim:.3f}')
    if chi_0 < chi_lim:
        print('Test passed')
    else:
        print('Test NOT passed')

    print('-------------------------------')
    print('Residuals')
    # print('     X       Y      Z')
    # print(f'{res[0]:8.3f} {res[1]:8.3f} {res[2]:8.3f}')
    for res in residuals:
        for dim in range(ndim):
            if dim == ndim-1:
                endline = '\n'
            else:
                endline = ' '
            print(f"{res[dim]:8.3f}", end=endline)

    print('-------------------------------')
    print(f'Covariance matrix:')
    for var in result.var_names:
        if var is result.var_names[-1]:
            endline = '\n'
        else:
            endline = ' '
        print(f"   {var:7s}", end=endline)

    for row in result.covar:
        for cov in row:
            if cov == row[-1]:
                endline = '\n'
            else:
                endline = ' '
            print(f"{cov:10.5f}", end=endline)
