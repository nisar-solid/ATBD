import numpy as np
import itertools
from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic


def remove_trend(x: np.ndarray,
                y: np.ndarray,
                data: np.ndarray) -> np.array:
    """This performs a basic 2d linear regression and removes the trend from
    the data. This is also known as 'de-ramping'.

    Parameters
    ----------
    x : np.ndarray
        x coordinates flattened
    y : np.ndarray
        y coordinates flattened
    data : np.ndarray
        Statistics/value to be de-trended

    Returns
    -------
    np.array
        The data with the fitted linear plane removed.
    """
    ones = np.ones(len(x))
    A = np.stack([x, y, ones], axis=1)
    ramp, _, _, _ = np.linalg.lstsq(A, data, rcond=None)

    new_data = data - A @ ramp
    return new_data


def get_emp_variogram(x: np.ndarray,
                      y: np.ndarray,
                      data: np.ndarray,
                      n_samples: int = None) -> tuple:
    """
    Obtains the (distances, empirical variogram) from de-trended data.

    Primary model assumptions are the mean is zero and the variance depends on
    space alone.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates
    y : np.ndarray
        The y-coordinate
    data : np.ndarray
        Statistic to construct variogram
    n_samples : int, optional
        How many samples to use (if None, then all samples used), by default
        None

    Returns
    -------
    tuple [np.ndarray, np.ndarray]
        (distances, variogram)
    """

    # Check if all the arrays are of the same shape
    assert((x.shape == y.shape) and
           (y.shape == data.shape)
           )

    x_, y_, data_ = x, y, data
    if n_samples is not None:
        n = len(x)
        samples = sorted(np.random.choice(n, n_samples, replace=False))
        x_ = x[samples]
        y_ = y[samples]
        data_ = data[samples]

    xy_pairs = np.stack([x_, y_], axis=1)
    distances = pdist(xy_pairs)

    combinations = list(itertools.combinations(range(len(x_)), 2))
    pair_ind_0, pair_ind_1 = zip(*combinations)

    data_0 = data_[list(pair_ind_0)]
    data_1 = data_[list(pair_ind_1)]

    # Dissimilarity
    # This is the empirical variance of zero mean data
    variogram = 0.5 * np.square(data_0 - data_1)

    return distances, variogram


def bin_variogram(distance: np.ndarray,
                  variogram: np.ndarray,
                  bins: int = 20,
                  distance_range: list = None) -> tuple:
    """Bin the distances and variogram for analysis

    Parameters
    ----------
    distance : np.ndarray
        Pairwise distances
    variogram : np.ndarray
        Variogram values associated with pairwise distances
    bins : int, optional
        Number of bins, by default 20
    distance_range : list, optional
        The range of distance to analyze (if None, whole range used), by default
        None.

    Returns
    -------
    tuple [np.ndarray, np.ndarray]
        (binned distances, binned variogram)
    """

    distance_range = distance_range or [0, np.nanmax(distance)]

    distance_binned, _, _ = binned_statistic(distance, distance,
                                             bins=bins, range=distance_range)
    variogram_binned, _, _ = binned_statistic(distance, variogram,
                                              bins=bins, range=distance_range)

    return distance_binned, variogram_binned
