import numpy as np
import math
import warnings
import copy

from solid_utils.variogram import remove_trend
from mintpy.utils import utils as ut


## Record measurement values at site locations
class SiteMeasurement:
    """Record the value of a measured parameter and the error on that
    measurement at a named location (site). Support different naming of that
    parameter via aliases (e.g., vel for velocity [Secular] and dis for
    displacement [Coseismic]).

    Differencing of two sites is defined as the subtraction of the parameter
    at the second site from that of the first. Error is propagated as the
    root sum of squares. Differencing can be performed by the subtraction
    operator "-", e.g., Site1 - Site2.
    """
    # Measurement alias
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, inp):
        self._x = inp

    @x.deleter
    def x(self):
        del self._x

    # Measurement error alias
    @property
    def x_err(self):
        return self._x_err

    @x_err.setter
    def x_err(self, inp):
        self._x_err = inp

    @x_err.deleter
    def x_err(self):
        del self._x_err

    def __init__(self, site:str, site_lon:float, site_lat:float,
                 x:float, x_err:float, unit:str):
        """Record the site name and geographic location, as well as the
        measurement value and error.

        Parameters: site  - str, site name, e.g., four-char GNSS station code
                    site_lon/lat - float, site geographic coordinates
                    x     - float, measured value at the site
                    x_err - float, stdev of measured value
                    unit  - str, measurement unit
        """
        # Site attributes
        self.site = site
        self.site_lon = site_lon
        self.site_lat = site_lat

        # Measurement attributes
        self.x = x
        self.x_err = x_err
        self.unit = unit

    def report(self, scale=1., print_unit=None):
        """Print the site name, measurement value, and uncertainty.
        """
        # Units to print
        if scale != 1. and print_unit is None:
            warnings.warn('Print scale was reset but units were not adjusted')
        print_unit = self.unit if print_unit is None else print_unit

        report_str = f"{self.site:s} " \
                + f"{scale*self.x:.2f} +- {scale*self.x_err:.2f} " \
                + f"{print_unit:s}"

        return report_str

    def __sub__(self, other):
        """Subtract the measured value of the "other" site from this site.
        Propagate the errors as the root sum of squares.
        Carry over the name and geographic location of the current site.

        Note that this definition overloads the subtraction operator.
        """
        # Check if units are the same between datasets
        if self.unit != other.unit:
            warnings.warn('Units do not match, continuing with differencing')

        # Create new object and carry over attributes
        resid = copy.deepcopy(self)

        # Subtract measurement values
        resid.x = self.x - other.x

        # Propagate error as root sum of squares
        resid.x_err = np.sqrt(self.x_err**2 + other.x_err**2)

        return resid


class SiteVelocity(SiteMeasurement):
    """Child class of SiteMeasurement specifically for velocity measurements.
    Inherits attributes of SiteMeasurement.
    """
    # Measurement information
    quantity = 'velocity'

    # Set aliases
    vel = SiteMeasurement.x
    vel_err = SiteMeasurement.x_err

    def __init__(self, site, site_lon, site_lat, vel, vel_err, unit='m/y'):
        super().__init__(site, site_lon, site_lat, vel, vel_err, unit)

class SiteDisplacement(SiteMeasurement):
    """Child class of SiteMeasurement specifically for displacement
    measurements.
    Inherits attributes of SiteMeasurement.
    """
    quantity = 'displacement'

    # Set aliases
    dis = SiteMeasurement.x
    dis_err = SiteMeasurement.x_err

    def __init__(self, site, site_lon, site_lat, dis, dis_err, unit='m'):
        super().__init__(site, site_lon, site_lat, dis, dis_err, unit)


## Collect samples from a raster dataset
# Seed random number generator for consistency
np.random.seed(1)

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

def load_geo_utm(attr_geo):
    """Produces X and Y vectors from geocoded files in UTM projection.
    Gives coordinates for middle of each pixel
    """
    X0=int(float(attr_geo['X_FIRST']))
    Y0=int(float(attr_geo['Y_FIRST']))
    X_step=int(float(attr_geo['X_STEP']))
    Y_step=int(float(attr_geo['Y_STEP']))
    length=int(attr_geo['LENGTH'])
    width=int(attr_geo['WIDTH'])

    
    X_start = X0+X_step/2
    Y_start = Y0+Y_step/2
    
    X = np.arange(X_start,X_start+X_step*width,X_step)
    Y = np.arange(Y_start,Y_start+Y_step*length,Y_step)
    
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


def samps_to_pairs(x,y,data):
    """Create pairs of data samples. If the lengths of x, y, and data
    is N, then each array will be formatted into an (N/2, 2) array.

    Parameters:
    x: np.ndarray
        input X location of the data points (1d)
    y: np.ndarray
        input Y location of the data points (1d)
    data: np.ndarray
        input data array (1d)

    Returns:
    x_pairs: np.ndarray
        pairs of x points (2d)
    y_pairs: np.ndarray
        pairs of y points (2d)
    data_pairs: np.ndarray
        pairs of data points (2d)
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

    x_pairs = np.column_stack([x_odd, x_even])
    y_pairs = np.column_stack([y_odd, y_even])
    data_pairs = np.column_stack([data_odd, data_even])

    return x_pairs, y_pairs, data_pairs


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
    # Pair up samples
    x_pairs, y_pairs, data_pairs = samps_to_pairs(x,y,data)

    # Parse paired sample coordinates
    x_odd = x_pairs[:,0]
    x_even = x_pairs[:,1]

    y_odd = y_pairs[:,0]
    y_even = y_pairs[:,1]

    data_odd = data_pairs[:,0]
    data_even = data_pairs[:,1]

    # Calculate distance between points
    distance = np.sqrt((x_odd-x_even)**2+(y_odd-y_even)**2)

    # Calculate difference between measurements
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


def profile_samples(x, y, data, metadata, len_rqmt, num_samples=10000):
    """Similar to samp_pair, randomly select data points and
    retrieve a profile (transection) between those points.
    Fit the data profile with a 1st order polynomial, and solve
    that for the predicted end points.

    Parameters:
    X: np.ndarray
        input longitudes of the data points (1d)
    Y: np.ndarray
        input latitudes of the data points(1d)
    data: np.ndarray
        input data array (1d)
    metadata: dict
        MintPy metadata describing spatial parameters of data set
    len_rqmt: list
        list of [<min>, <max>]-allowable profile distances, per ATBD
        requirement
    num_samples: int
        number of points to be sampled, defalt value is 10,000

    Returns:
    distance: np.ndarray
        distances for every data pair (1d)
    rel_measure: np.ndarray
        absolute value of data difference for every data pair (1d)
    """
    # mask nan, result in 1d
    data1d = data.reshape(-1,1)
    mask = np.isnan(data1d)
    data1d = data1d[~mask]
    x1d = x[~mask]
    y1d = y[~mask]

    # Draw random samples based on geographic coordinates
    data1d, x1d, y1d = rand_samp(data1d, x1d, y1d, 2*num_samples)

    # Reshape sample points into pairs (x1, x2), (y1, y2)
    x_pairs, y_pairs, data_pairs = samps_to_pairs(x1d, y1d, data1d)

    # Determine distances between pairs to check against length requirements
    dists = haversine_distance(\
                                x_pairs[:,0], y_pairs[:,0],
                                x_pairs[:,1], y_pairs[:,1]\
                                )
    valid_indices = (dists > len_rqmt[0]) & (dists < len_rqmt[1])

    # Filter for data within length requirements
    x_pairs = x_pairs[valid_indices]
    y_pairs = y_pairs[valid_indices]
    data_pairs = data_pairs[valid_indices]

    # Number of valid sample points
    num_valid_samples = np.sum(valid_indices)

    print('{:d} / {:d} profiles have valid lengths (0.1 - 50 km)'.\
        format(num_valid_samples, num_samples))

    # Loop through sample pairs
    dist = np.empty(num_valid_samples)
    rel_measure = np.empty(num_valid_samples)
    for i in range(num_valid_samples):
        # Retreive transect
        start_lalo = [y_pairs[i,0], x_pairs[i,0]]
        end_lalo = [y_pairs[i,1], x_pairs[i,1]]
        transect = ut.transect_lalo(data, metadata, start_lalo, end_lalo,
                interpolation='nearest')

        # Convert distance from m to km
        transect['distance'] /= 1000

        # Solve for fit
        if transect['distance'].size == 0 or transect['distance'].size == 0:
            # Store distance and difference values
            dist[i] = np.nan
            rel_measure[i] = np.nan

            continue

        fit = np.polyfit(transect['distance'], transect['value'], 1)

        # Solve for values at profile start and end points
        endpoint_data = np.poly1d(fit)([0, transect['distance'].max()])

        # Compute abosolute difference between end points
        abs_measurement_diff = np.abs(np.diff(endpoint_data)[0])

        # Store distance and difference values
        dist[i] = transect['distance'].max()
        rel_measure[i] = abs_measurement_diff

    return dist, rel_measure


def haversine_distance(start_lon, start_lat, end_lon, end_lat):
    """Compute the distance between two points geographic points using
    the haversine formula.
    """
    # Convert coordinate degrees to radians
    start_lat = np.deg2rad(start_lat)
    start_lon = np.deg2rad(start_lon)

    end_lat = np.deg2rad(end_lat)
    end_lon = np.deg2rad(end_lon)

    # Apply haversine formula
    delta_lat = end_lat - start_lat
    delta_lon = end_lon - start_lon

    a = np.sin(delta_lat/2)**2 \
        + np.cos(start_lat) \
        * np.cos(end_lat) \
        * np.sin(delta_lon/2)**2

    c = 2 * np.arcsin(np.sqrt(a))
    r = 6378  # Earth's radius

    return c * r
