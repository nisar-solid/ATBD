#permafrost utilities
from pathlib import Path
from osgeo import gdal
from typing import List, Union

import abc
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, namedtuple
import h5py
import matplotlib
import matplotlib.pyplot as plt


def get_common_overlap(file_list: List[Union[str, Path]]) -> List[float]:
    """Get the common overlap of  a list of GeoTIFF files
    
    Arg:
        file_list: a list of GeoTIFF files
    
    Returns:
         [ulx, uly, lrx, lry], the upper-left x, upper-left y, lower-right x, and lower-right y
         corner coordinates of the common overlap.
         
    from: https://github.com/ASFHyP3/hyp3-docs/blob/develop/docs/tutorials/hyp3_insar_stack_for_ts_analysis.ipynb
    """
    
    corners = [gdal.Info(str(dem), format='json')['cornerCoordinates'] for dem in file_list]

    ulx = max(corner['upperLeft'][0] for corner in corners)
    uly = min(corner['upperLeft'][1] for corner in corners)
    lrx = min(corner['lowerRight'][0] for corner in corners)
    lry = max(corner['lowerRight'][1] for corner in corners)
    return [ulx, uly, lrx, lry]

def clip_hyp3_products_to_common_overlap(data_dir: Union[str, Path], overlap: List[float]) -> None:
    """Clip all GeoTIFF files to their common overlap
    
    Args:
        data_dir:
            directory containing the GeoTIFF files to clip
        overlap:
            a list of the upper-left x, upper-left y, lower-right-x, and lower-tight y
            corner coordinates of the common overlap
    Returns: None
    
    from: https://github.com/ASFHyP3/hyp3-docs/blob/develop/docs/tutorials/hyp3_insar_stack_for_ts_analysis.ipynb
    """

    
    files_for_mintpy = ['_water_mask.tif', '_corr.tif', '_unw_phase.tif', '_dem.tif', '_lv_theta.tif', '_lv_phi.tif']

    for extension in files_for_mintpy:

        for file in data_dir.rglob(f'*{extension}'):

            dst_file = file.parent / f'{file.stem}_clipped{file.suffix}'

            gdal.Translate(destName=str(dst_file), srcDS=str(file), projWin=overlap)

def get_watermask(hyp3_dir):
    watermaskfile = next(hyp3_dir.glob(f'*/*_water_mask_clipped.tif'))
    return gdal.Open(str(watermaskfile)).ReadAsArray()
    
class DeformationModel(abc.ABC):
    """Spline model reconstruction method from:
            Zwieback, S., & Meyer, F. J. (2021). Top-of-permafrost ground ice 
            indicated by remotely sensed late-season subsidence. The Cryosphere, 
            15(4), 2041-2055."""

    def fit(self, timeseries):
        res = self._fit(timeseries.timeseries, dates_o=timeseries.dates)
        return res
    
    def _fit(self, timeseriesarr, dates_o):
        res = defaultdict(lambda: None) # change this to a proper object
        res['A'] = self.design_matrix(dates=dates_o)
        res['b'] = np.einsum('ij, jkl -> ikl', np.linalg.pinv(res['A']), timeseriesarr)
        res['dates_o'] = dates_o
        return res

    def reconstruct(self, timeseries=None, dates_r=None, res=None, first_zero=True):
        if res is None:
            res = self.fit(timeseries)
        if dates_r is None:
            dates_r = res['dates_o']
            A = res['A']
        else:
            A = self.design_matrix(dates=dates_r)
        rec = self._reconstruct(A, res['b'], first_zero=first_zero)
        return Timeseries(rec, dates_r, meta=timeseries.meta)
        
    def _reconstruct(self, A, b, first_zero=True):

        timeseries_rec = np.einsum('ij, jkl -> ikl', A, b)

        if first_zero:
            timeseries_rec -= timeseries_rec[0, ...][np.newaxis, ...]
        return timeseries_rec
        
    
class SplineModel(DeformationModel):
    """Develops model of deformation using spline fitting. From Zwieback and Meyer (2021)."""

    def __init__(self, dates_o=None, spacing=42.0, relbuf=0.5):
        # spacing: node spacing in days
        # relbuf: relative buffer so that the time between the first and last node is (1 + relbuf) * observation period
        self.spacing = spacing if spacing is not None else spacing_def
        self.relbuf = relbuf if relbuf is not None else relbuf_def
        if dates_o is not None:
            self.init_dates(dates_o)
        else:
            self.nodes = None
    
    def init_dates(self, dates_o):
        self.dates_o = dates_o
        self.nodes = self._nodes(dates_o)
        
    def design_matrix(self, dates=None):
        # dates: iterable containing dates (datetime objects); defaults to observation dates used in initialization
        if self.nodes is None: 
            raise ValueError('Model needs to be initialized with init_dates')
        if dates is None:
            dates = self.dates_o
        x_n, x = self._x(self.nodes), self._x(dates)

        A_raw = np.stack([self._bspline_ad(x, x_n_) for x_n_ in x_n], axis=1)
        A = A_raw - np.mean(A_raw, axis=0)[np.newaxis, :]
        return A
        
    def _nodes(self, dates_o):
        l = (max(dates_o) - min(dates_o)).days * (1 + self.relbuf)
        N = int(np.floor(l / self.spacing))
        node0 = min(dates_o) + 0.5 * (max(dates_o) - min(dates_o)) - (N - 1) / 2 * timedelta(days=self.spacing)
        nodes = tuple([node0 + timedelta(days=n * self.spacing) for n in range(N)])
        return nodes

    def _x(self, dates):
        # scaled, dimension-free coordinates so that spline node spacing is 1.0
        return np.array([(d - self._dateref).days / self.spacing for d in dates])
    
    @property
    def _dateref(self):
        # for internal bookkeeping (definition of x)
        return self.nodes[len(self.nodes)//2]
   
    def _bspline(self, x, xn, scal=1.0):
        # basic quadratic cardinal B-spline without boundary knots: used for the deformation rate
        # the x coordinates are assumed normalized (for scal = 1), i.e. the node spacing is 1.0 on the x scale
        nd = (x - xn) / scal
        y = np.zeros_like(nd)
        y += ((1 / 2) * nd ** 2 + (3 / 2) * nd + (9 / 8)) * np.logical_and(nd >= -1.5, nd < -0.5)
        y += (-nd ** 2 + (3 / 4)) * np.logical_and(nd >= -0.5, nd < 0.5)
        y += ((1 / 2) * nd ** 2 - (3 / 2) * nd + (9 / 8)) * np.logical_and(nd >= 0.5, nd < 1.5)
        return y

    def _bspline_ad(self, x, xn, scal=1.0):
        # antiderivative of the quadratic bspline: used for deformation
        # the x coordinates are assumed normalized (for scal = 1), i.e. the node spacing is 1.0 on the x scale
        nd = (x - xn) / scal
        y = np.zeros_like(nd)
        y += ((1 / 6) * nd ** 3 + (3 / 4) * nd ** 2 + (9 / 8) * nd + (9 / 16)) * np.logical_and(nd >= -1.5, nd < -0.5)
        y += (-(1 / 3) * nd ** 3 + (3 / 4) * nd + (1 / 2)) * np.logical_and(nd >= -0.5, nd < 0.5)
        y += ((1 / 6) * nd ** 3 - (3 / 4) * nd ** 2 + (9 / 8) * nd + (7 / 16)) * np.logical_and(nd >= 0.5, nd < 1.5)
        y += 1.0 * (nd >= 1.5)
        return y
    
class Timeseries():
    
    def __init__(self, ts, dates, meta=None,work_dir=None):
        self.timeseries = ts
        self.dates = dates
        self.meta = meta
        if work_dir==None:
            work_dir = Path.cwd()
        self.work_dir = work_dir
    
    @classmethod
    def from_file(cls, fn='default', mask=None):
        if fn=='default':
            fn = self.work_dir/'timeseries.h5'
        with h5py.File(fn, 'r') as f:
            datesb = list(f['date'])
            dates_o = [datep(d) for d in datesb]
            timeseries = np.array(f['timeseries'])
            if mask is not None:
                timeseries[:, mask] = np.nan
        meta = read_metadata(fn)
        return cls(timeseries, dates_o, meta=meta)

def datep(dateb):
    return datetime.strptime(str(int(dateb)), '%Y%m%d') 

def read_metadata(fntimeseries):
    with h5py.File(fntimeseries, 'r') as f:
        metadata = dict(f.attrs)
    return metadata

def indices_geolocation(xy, meta):
    geotrans = (float(meta['X_FIRST']), float(meta['X_STEP']), 0.0, float(meta['Y_FIRST']), 0.0, float(meta['Y_STEP']))
    ind0 = (xy[1] - geotrans[0]) / geotrans[1]
    ind1 = (xy[0] - geotrans[3]) / geotrans[5]
    return (int(ind1), int(ind0))

def plot_point(coords, timeseries, timeseries_marker=None):
    ind = indices_geolocation(coords, timeseries.meta)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(
        timeseries.dates, timeseries.timeseries[:, ind[0], ind[1]], c='#666666')
    if timeseries_marker is not None:
        ax.plot(
            timeseries_marker.dates, timeseries_marker.timeseries[:, ind[0], ind[1]],  
            marker='o',linestyle='none', mfc='none', mec='#666666')
    formatter = matplotlib.dates.DateFormatter('%b %d') 
    ax.xaxis.set_major_formatter(formatter) 
    fig.suptitle(f'{coords[0]:6.2f} {coords[1]:6.2f}: {timeseries.dates[0].year} seasonal deformation')

def plot_basis_functions(dm, dates=None):
    if dates is None: dates = dm.dates_o
    dates_ = [dates[0] + timedelta(days=d) for d in range((dates[-1] - dates[0]).days)]
    A = dm.design_matrix(dates=dates_)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    Nb = A.shape[1]
    for jb in range(Nb):
        c = jb / (Nb - 0.2)
        ax.plot(dates_, A[:, jb], c=(c,) * 3)
    formatter = matplotlib.dates.DateFormatter('%b %d') 
    ax.xaxis.set_major_formatter(formatter) 
    fig.suptitle(f'{dates_[0].year} deformation basis functions')
                     
if __name__ == '__main__':
    pass