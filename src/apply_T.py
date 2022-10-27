import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from lmfit import Minimizer, Parameters

from lib.rotra import compute_residuals, compute_approx_values
from lib.io import read_data_to_df
from lib.utils import print_results

pt_in = read_data_to_df(
    'data/.txt',  # 'data/belpy/loc.txt',
    delimiter=',',
    header=None,
    col_names=['x', 'y', 'z'],
    index_col=0,
)

print('Point world:\n', pt_world)
print('Point loc:\n', pt_loc)
