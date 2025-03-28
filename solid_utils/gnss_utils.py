import numpy as np
import math
import warnings

from solid_utils.variogram import remove_trend
from mintpy.utils import time_func, utils as ut

# def retrieve_gnss_component(gnss_stn, component:str):
#     """Retrieve component of motion.

#     Parameters: gnss_stn  - GNSS obj, GNSS data with displacement time-series
#                 component - str, component of motion (E, N, U, or LOS)
#     Returns:    dis       - np.ndarray, displacement time-series
#     """
#     # Check component specified
#     component = component.upper()
#     if component not in ['E', 'N', 'U', 'LOS']:
#         raise Exception('Component of motion must be one of E, N, U, or LOS')

#     # Retrieve component of motion
#     if component == 'E':
#         return gnss_stn.dis_e
#     elif component == 'N':
#         return gnss_stn.dis_n
#     elif component == 'U':
#         return gnss_stn.dis_u
#     elif component == 'LOS':
#         return gnss_stn.dis_los

# def model_gnss_timeseries(gnss_stn, component:str, model:dict):
#     """Model a GNSS displacement time-series.

#     Parameters: gnss_stn  - GNSS obj, GNSS data with displacement time-series
#                 component - str, component of motion (E, N, U, or LOS)
#                 model     - dict, time function model
#     Returns:    dis_hat   - np.ndarray, array of predicted displacement values
#                 mhat      - np.ndarray, model fit parameters
#                 mhat_se   - np.ndarray, standard errors for model fit params
#     """
#     # Construct design matrix from dates and model
#     date_list = [date.strftime('%Y%m%d') for date in gnss_stn.dates]
#     G = time_func.get_design_matrix4time_func(date_list, model)

#     # Invert for model parameters
#     dis = retrieve_gnss_component(gnss_stn, component)
#     m_hat = np.linalg.pinv(G).dot(dis)

#     # Predict displacements
#     dis_hat = np.dot(G, m_hat)

#     # Quantify error on model parameters
#     resids = dis - dis_hat
#     sse = np.sum(resids**2)
#     n = len(dis_hat)
#     dof = len(m_hat)
#     c = sse/(n-dof) * np.linalg.inv(np.dot(G.T, G))
#     mhat_se = np.sqrt(np.diag(c))

#     return dis_hat, m_hat, mhat_se

# def modify_gnss_series(gnss_stn, remove_ndxs):
#     """Remove dates from all components of a GNSS time-series based on a
#     logical array.

#     Parameters: gnss_stn    - GNSS obj, GNSS data with displacement time-series
#                 remove_ndxs - np.ndarray, boolean array where True indicates a
#                               value to remove
#     Returns:    gnss_stn    - GNSS obj, modified GNSS data
#     """
#     gnss_stn.dates = gnss_stn.dates[~remove_ndxs]
#     gnss_stn.dis_e = gnss_stn.dis_e[~remove_ndxs]
#     gnss_stn.dis_n = gnss_stn.dis_n[~remove_ndxs]
#     gnss_stn.dis_u = gnss_stn.dis_u[~remove_ndxs]
#     gnss_stn.std_e = gnss_stn.std_e[~remove_ndxs]
#     gnss_stn.std_n = gnss_stn.std_n[~remove_ndxs]
#     gnss_stn.std_u = gnss_stn.std_u[~remove_ndxs]
#     if hasattr(gnss_stn, 'dis_los'):
#         gnss_stn.dis_los = gnss_stn.dis_los[~remove_ndxs]
#         gnss_stn.std_los = gnss_stn.std_los[~remove_ndxs]

#     return gnss_stn

# def outliers_zscore(dis:np.ndarray, dis_hat:np.ndarray, threshold:float):
#     """Identify outliers using the z-score metric.

#     Compute the number of standard deviations the data are from the mean
#     and return the indices of values greater than the specified threshold.

#     Parameters: dis          - np.ndarray, array of displacement values
#                 dis_hat      - np.ndarray, array of predicted displacement values
#                 threshold    - float, z-score value (standard deviation)
#                                beyond which to exclude data
#     Returns:    outlier_ndxs - np.ndarray, boolean array where True
#                                indicates an outlier
#                 n_outliers   - int, number of outliers
#     """
#     zscores = (dis - dis_hat) / np.std(dis)
#     outlier_ndxs = np.abs(zscores) > threshold
#     n_outliers = np.sum(outlier_ndxs)

#     return outlier_ndxs, n_outliers

# def remove_gnss_outliers(gnss_stn, component:str, model:dict,
#                          threshold=3, max_iter=2, verbose=False):
#     """Determine which data points are outliers based on the z-score
#     metric and remove those points.

#     Parameters: gnss_stn  - GNSS obj, GNSS data with displacement time-series
#                 component - str, component of motion (E, N, U, or LOS)
#                 model     - dict, time function model
#                 threshold - float, standard deviations beyond which values
#                             are considered outliers
#                 max_iter  - int, maximutm number of iterations before stopping
#     Returns:    gnss_stn  - GNSS obj, GNSS data with outliers removed
#     """
#     if verbose == True:
#         print('Station {:s} original data set size: {:d}'.\
#               format(gnss_stn.site, len(gnss_stn.dates)))

#     # Retrieve observed values and predict values based on model
#     dis = retrieve_gnss_component(gnss_stn, component)
#     dis_hat, _, _ = model_gnss_timeseries(gnss_stn, component, model)

#     # Determine outliers based on z-score
#     outlier_ndxs, n_outliers = outliers_zscore(dis, dis_hat, threshold)

#     # Initialize counter
#     i = 0

#     # Remove outliers from data set
#     while (n_outliers > 0) or (i < max_iter):
#         if verbose == True:
#             print(f'nb outliers: {n_outliers:d}')

#         # Update time series
#         modify_gnss_series(gnss_stn, outlier_ndxs)

#         # Update number of outliers
#         dis = retrieve_gnss_component(gnss_stn, component)
#         dis_hat, _, _ = model_gnss_timeseries(gnss_stn, component, model)
#         outlier_ndxs, n_outliers = outliers_zscore(dis, dis_hat, threshold)

#         # Update counter
#         i += 1

#     if verbose == True:
#         print('Station {:s} final data set size: {:d}'.\
#               format(gnss_stn.site, len(gnss_stn.dates)))

#     return gnss_stn
