# Author: R Zinke
# March, 2025

import numpy as np
from scipy import stats

from mintpy.utils import time_func, utils as ut


class TSModelFit:
    def __init__(self, dates:np.ndarray, dis:np.ndarray, model:dict, conf=95.45):
        """Model a displacement time-series.
    
        Parameters: dates     - np.ndarray, dates as Python datetime objects
                    dis       - np.ndarray, displacements
                    model     - dict, time function model
        Attributes: dis_hat   - np.ndarray, array of predicted displacement
                                values
                    m_hat     - np.ndarray, model fit parameters
                    mhat_se   - np.ndarray, standard errors for model fit params
                    err_envelope - list of np.ndarray, upper and lower confidence
                                bounds
        """
        # Record parameters
        self.model = model
        if len(dates) != len(dis):
            raise ValueError('Must have same number of dates and displacements')
        self.dates = dates
        self.dis = dis
        self.n = len(dates)

        # Construct design matrix from dates and model
        date_list = [date.strftime('%Y%m%d') for date in dates]
        G = time_func.get_design_matrix4time_func(date_list, self.model)
    
        # Invert for model parameters
        self.m_hat = np.linalg.pinv(G).dot(self.dis)
    
        # Predict displacements
        self.dis_hat = np.dot(G, self.m_hat)
    
        # Quantify error on model parameters
        resids = self.dis - self.dis_hat
        sse = np.sum(resids**2)
        dof = len(self.m_hat)
        pcov = sse/(self.n - dof) * np.linalg.inv(np.dot(G.T, G))
        self.mhat_se = np.sqrt(np.diag(pcov))
    
        # Propagate uncertainty
        dcov = G.dot(pcov).dot(G.T)
        derr = np.sqrt(np.diag(dcov))
    
        # Error envelope
        err_scale = stats.t.interval(conf/100, dof)
        err_lower = self.dis_hat + err_scale[0] * derr
        err_upper = self.dis_hat + err_scale[1] * derr
        self.err_envelope = [err_lower, err_upper]


class IterativeOutlierFit:
    @staticmethod
    def outliers_zscore(dis:np.ndarray, dis_hat:np.ndarray, threshold:float):
        """Identify outliers using the z-score metric.
    
        Compute the number of standard deviations the data are from the mean
        and return the indices of values greater than the specified threshold.
    
        Parameters: dis          - np.ndarray, array of displacement values
                    dis_hat      - np.ndarray, array of predicted displacement
                                   values
                    threshold    - float, z-score value (standard deviation)
                                   beyond which to exclude data
        Returns:    outlier_ndxs - np.ndarray, boolean array where True
                                   indicates an outlier
                    n_outliers   - int, number of outliers
        """
        zscores = (dis - dis_hat) / np.std(dis - dis_hat)
        outlier_ndxs = np.abs(zscores) > threshold
        n_outliers = np.sum(outlier_ndxs)
    
        return outlier_ndxs, n_outliers
    
    def __init__(self, dates, dis, model, threshold=3, max_iter=2):
        """Determine which data points are outliers based on the z-score
        metric and remove those points.

        Parameters: dates     - np.ndarray, dates as Python datetime objects
                    dis       - np.ndarray, displacements
                    model     - dict, time function model
                    threshold - float, standard deviations beyond which values
                                are considered outliers
                    max_iter  - int, maximutm number of iterations before
                                stopping
        Attributes: dis_hat   - np.ndarray, array of predicted displacement
                                values
                    mhat      - np.ndarray, model fit parameters
                    mhat_se   - np.ndarray, standard errors for model fit
                                params
                    err_envelope - list of np.ndarray, upper and lower confidence
                                bounds
        """
        # Record parameters
        self.dates = dates
        self.dis = dis
        self.model = model
        self.outlier_threshold = threshold

        # Initialize outlier removal
        self.iters = 0
        self.outlier_dates = np.array([])
        self.outlier_dis = np.array([])

        # Initial fit to data
        ts_model = TSModelFit(self.dates, self.dis, self.model)

        # Determine outliers based on z-score
        (outlier_ndxs,
         n_outliers) = self.outliers_zscore(ts_model.dis, ts_model.dis_hat,
                                            self.outlier_threshold)
        self.n_outliers = n_outliers

        # Remove outliers from data set
        while (n_outliers > 0) and (self.iters < max_iter):
            # Update time series
            self.outlier_dates = np.append(self.outlier_dates,
                                           self.dates[outlier_ndxs])
            self.outlier_dis = np.append(self.outlier_dis,
                                         self.dis[outlier_ndxs])

            self.dates = self.dates[~outlier_ndxs]
            self.dis = self.dis[~outlier_ndxs]

            # Update timeseries model
            ts_model = TSModelFit(self.dates, self.dis, self.model)

            # Determine outliers based on z-score
            (outlier_ndxs,
             n_outliers) = self.outliers_zscore(ts_model.dis, ts_model.dis_hat,
                                                self.outlier_threshold)
            self.n_outliers += n_outliers

            # Update iteration counter
            self.iters += 1

        # Record final parameters
        self.m_hat = ts_model.m_hat
        self.mhat_se = ts_model.mhat_se
        self.dis_hat = ts_model.dis_hat
        self.err_envelope = ts_model.err_envelope
