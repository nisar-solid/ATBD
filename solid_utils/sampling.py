import numpy as np
import math
import warnings
from solid_utils.variogram import remove_trend


def load_geo(attr_geo):
    """This program calculate the coordinate of the geocoded files
    and perform coordinate transformation from longitude and latitude to local coordinate in kilometers.

    The coordinate transformation is done with several assumption.
    1.) The earth is a sphere with radius equals to 6371 km. So both of the distance of one latitude degree
    and the distance of one longitude degree at equator is 2*PI*6371/360 = 111.195 km.
    2.) The distance of one longitude degreevaries with latitude. Here it is simply assumed to be the one
    at the central latitude of the input scene.

    Parameters:
    geo_attr:attribute of the geocoded data

    Returns:
    X:coordinates in east direction in km
    Y:coordinates in north direction in km
    """
    Y0=float(attr_geo['Y_FIRST'])
    X_step=float(attr_geo['X_STEP'])
    Y_step=float(attr_geo['Y_STEP'])
    length=int(attr_geo['LENGTH'])
    width=int(attr_geo['WIDTH'])
    
    Y_local_end_ix = math.floor(length/2)
    Y_local_first_ix = -(length-Y_local_end_ix-1)
    X_local_end_ix = math.floor(width/2)
    X_local_first_ix = -(width-X_local_end_ix-1)
    
    Y_origin = Y0+Y_step*(-Y_local_first_ix)
    
    X_step_local = math.cos(math.radians(Y_origin))*X_step*111.195
    Y_step_local = Y_step*111.195
    

    X=np.linspace(X_local_first_ix*X_step_local,X_local_end_ix*X_step_local,width)
    Y=np.linspace(Y_local_first_ix*Y_step_local,Y_local_end_ix*Y_step_local,length)

    return X,Y

def rand_samp(data:np.ndarray,X:np.ndarray,Y:np.ndarray,num_samples:int=10000):
    """This function randomly selects data points, all data points will be used if
    num_samples > len(data).

    Parameters:
    data: np.ndarray
        input data array (1d)
    X: np.ndarray
        input X location of the data points (1d)
    Y: np.ndarray
        input Y location of the data points (1d)
    num_samples: int
        number of points to be sampled

    Returns:
    sampled_data: np.ndarray
        sampled data array (1d)
    sampled_X: np.ndarray
        X location of the sampled data points (1d)
    sampled_Y: np.ndarray
        Y location of the sampled data points (1d)
    """
    length=np.size(data)
    if length<num_samples:
        n_points=length
        warnings.warn(f'Using all data points: {n_points}')
    else:
        n_points=num_samples
        
    ind=np.random.choice(length,n_points,replace=False)
    sampled_data=data[ind]
    sampled_X=X[ind]
    sampled_Y=Y[ind]

    return sampled_data, sampled_X, sampled_Y

def pair_up(x,y,data):
    """Pair up data samples.

    Parameters:
    x: np.ndarray
        input X location of the data points (1d)
    y: np.ndarray
        input Y location of the data points (1d)
    data: np.ndarray
        input data array (1d)

    Returns:
    distance: np.ndarray
        distances for every data pairs (1d)
    rel_measure: np.ndarray
        absolute value of data difference for every data pairs (1d)
    """
    x_odd = x[1::2]
    y_odd = y[1::2]
    data_odd = data[1::2]
    if (x.shape[0] % 2) == 0:
        x_even = x[0::2]
        y_even = y[0::2]
        data_even = data[0::2]
    else:
        x_even = x[0:-1:2]
        y_even = y[0:-1:2]
        data_even = data[0:-1:2]
    distance = np.sqrt((x_odd-x_even)**2+(y_odd-y_even)**2)
    rel_measure = abs(data_odd-data_even)
    return distance,rel_measure

def samp_pair(x,y,data,num_samples=1000000,deramp=False):
    """Randomly select data points and pair up them.
    This function is based on rand_samp and pair_up.
    all data points will be used if num_samples > len(data).
    This function also provide option to remove trend.

    Parameters:
    data: np.ndarray
        input data array (1d)
    X: np.ndarray
        input X location of the data points (1d)
    Y: np.ndarray
        input Y location of the data points (1d)
    num_samples: int
        number of points to be sampled, default value is 1000000
    deramp: Bool
        flag to remove trend, default value is False

    Returns:
    distance: np.ndarray
        distances for every data pairs (1d)
    rel_measure: np.ndarray
        absolute value of data difference for every data pairs (1d)
    """
    
    assert x.shape == y.shape
    assert x.shape == data.shape
    assert x.ndim == 2
        
    # mask nan, result in 1d
    mask = np.isnan(data)
    data1d = data[~mask]
    x1d = x[~mask]
    y1d = y[~mask]
    
    # remove trend (optional)
    if deramp:
        data1d = remove_trend(x1d,y1d,data1d)
    
    # randomly sample
    data1d,x1d,y1d = rand_samp(data1d,x1d,y1d,num_samples=num_samples)
    
    # pair up
    distance, rel_measure = pair_up(x1d,y1d,data1d)
    return distance, rel_measure
