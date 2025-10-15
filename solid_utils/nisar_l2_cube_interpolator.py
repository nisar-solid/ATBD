
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/23/2024

@author: Xiaodong Huang @JPL Caltech
"""

import argparse
import os
import time
import warnings

import h5py
import numpy as np
import rasterio
from osgeo import gdal, osr
from rasterio.coords import BoundingBox
from rasterio.warp import transform_bounds
from scipy.interpolate import RegularGridInterpolator

# GCOV datasets in its radar grid datacube
GCOV_RADAR_GRID_DATASETS = ['alongTrackUnitVectorX',
                            'alongTrackUnitVectorY',
                            'elevationAngle',
                            'groundTrackVelocity',
                            'incidenceAngle',
                            'losUnitVectorX',
                            'losUnitVectorY',
                            'zeroDopplerAzimuthTime',
                            'slantRange']

# GSLC datasets in its radar grid datacube
GSLC_RADAR_GRID_DATASETS = GCOV_RADAR_GRID_DATASETS

# GOFF datasets in its radar grid datacube
GOFF_RADAR_GRID_DATASETS = \
    GCOV_RADAR_GRID_DATASETS[:-2] + ['parallelBaseline',
                                     'perpendicularBaseline',
                                     'referenceSlantRange',
                                     'referenceZeroDopplerAzimuthTime',
                                     'secondarySlantRange',
                                     'secondaryZeroDopplerAzimuthTime']

# GUNW datasets in its radar grid datacube
GUNW_RADAR_GRID_DATASETS = \
    GOFF_RADAR_GRID_DATASETS + ['hydrostaticTroposphericPhaseScreen',
                                'wetTroposphericPhaseScreen',
                                'combinedTroposphericPhaseScreen',
                                'slantRangeSolidEarthTidesPhase']

def get_parser():
    '''
    Command line parser.
    '''
    descr = 'Interpolate the NISAR L2 3D metadata datacube'
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument(type=str,
                        dest='input_file',
                        help='Input NISAR L2 file')

    parser.add_argument('--dem',
                        '--dem-file',
                        dest='input_dem_file',
                        required=True,
                        type=str,
                        help='Reference DEM file')

    parser.add_argument('--out',
                        '--output',
                        dest='output_file',
                        type=str,
                        required=True,
                        help='Output file of the interplated 2D dataset')

    parser.add_argument('--frequency',
                        '--freq',
                        type=str,
                        default='A',
                        dest='frequency',
                        choices=['A', 'B'],
                        help='Frequency band, default: A')

    parser.add_argument('--polarization',
                        '--pol',
                        type=str,
                        default='HH',
                        dest='polarization',
                        choices=['HH', 'HV', 'VH', 'VV'],
                        help='Polarizations, default: HH')

    parser.add_argument('--cube-interp-method',
                        dest='cube_interp_method',
                        type=str,
                        default='linear',
                        choices=['linear', 'nearest', 'slinear', 'cubic',
                                 'quintic', 'pchip'],
                        help='Datacube interpolation method'
                        ' of the python RegularGridInterpolator, default: linear')

    parser.add_argument('--dem-resample-method',
                        dest='dem_resample_method',
                        type=str,
                        default='cubicspline',
                        choices=['near', 'bilinear', 'cubic', 'cubicspline',
                                 'lanczos', 'average','med'],
                        help='DEM interpolation method, default: cubicspline')

    parser.add_argument('--out-resampled-dem',
                        '--out-dem',
                        type=str,
                        default=None,
                        dest='out_resampled_dem',
                        help='Output of the resampled dem, default: None')

    parser.add_argument('--out-format',
                        dest='out_format',
                        default='GTiff',
                        help="The raster format of the output,"
                        " must be the GDAL supported raster format, default: GTiff")

    parser.add_argument('--force-resample-dem',
                        action='store_true',
                        dest='force_resample_dem',
                        help='Force resample dem')

    parser.add_argument('--gunw-ds-name',
                        dest='gunw_dataset_name',
                        default='unwrappedInterferogram',
                        choices=['unwrappedInterferogram',
                                 'wrappedInterferogram',
                                 'pixelOffsets'],
                        help='GUNW dataset names, default: unwrappedInterferogram')

    parser.add_argument('--cube-ds-name',
                        dest='cube_dataset_name',
                        required=True,
                        default='incidenceAngle',
                        choices=GUNW_RADAR_GRID_DATASETS + ['zeroDopplerAzimuthTime',
                                                            'slantRange'],
                        help='3D metadata datacube names')
    return parser.parse_args()


def compute_coverage_area(dem_bounds, ds_bounds):
    x_min = max(dem_bounds.left, ds_bounds.left)
    y_min = max(dem_bounds.bottom, ds_bounds.bottom)
    x_max = min(dem_bounds.right, ds_bounds.right)
    y_max = min(dem_bounds.top, ds_bounds.top)

    if x_min < x_max and y_min < y_max:
        return (x_max - x_min) * (y_max - y_min) / \
            ((ds_bounds.right -  ds_bounds.left)*\
                (ds_bounds.top -  ds_bounds.bottom)) * 100.0
    else:
        return 0.0

def resample_dem(input_dem_path,
                 out_epsg,
                 out_start_x,
                 out_start_y,
                 out_spacing_x,
                 out_spacing_y,
                 out_length,
                 out_width,
                 force_resample = False,
                 dem_resample_method = 'cubicspline',
                 out_path = None):

    # Check the DEM coverage over the dataset
    # raise warning if the coverage is less than 100.0
    with rasterio.open(input_dem_path) as src:
        dem_bounds = src.bounds
        original_crs = src.crs
        target_crs = f'EPSG:{out_epsg}'

        # DEM bounds to the target CRS
        dem_bounds = \
            transform_bounds(original_crs, target_crs,
                             dem_bounds.left, dem_bounds.bottom,
                             dem_bounds.right, dem_bounds.top)
        dem_bounds = BoundingBox(left = dem_bounds[0],
                                 bottom = dem_bounds[1],
                                 right = dem_bounds[2],
                                 top = dem_bounds[3])
        # the dataset bounds
        ds_bounds = BoundingBox(left=out_start_x,
                                bottom=out_start_y + out_length * out_spacing_y,
                                right=out_start_x + out_width * out_spacing_x,
                                top=out_start_y)
        coverage_area = round(compute_coverage_area(dem_bounds, ds_bounds),2)
        if coverage_area < 100.0:
            warnings.warn(
                f"DEM only covers {coverage_area}% of the dataset!",
                UserWarning)

    # If the DEM exists and the resample is not forced,
    # then read it from the existing file
    if ((out_path is not None) and
        (os.path.exists(out_path)) and
        (not force_resample)):
        des = gdal.Open(out_path)
        band = des.GetRasterBand(1)
        resampled_dem = band.ReadAsArray()
    else:
        # Resample the DEM using GDALWarp
        input_raster = gdal.Open(input_dem_path)
        output_extent = (out_start_x,
                         out_start_y + out_length * out_spacing_y,
                         out_start_x + out_width * out_spacing_x,
                         out_start_y)

        gdalwarp_options = gdal.WarpOptions(format="MEM",
                                            outputType = gdal.GDT_Float32,
                                            dstSRS=f"EPSG:{out_epsg}",
                                            xRes=out_spacing_x,
                                            yRes=np.abs(out_spacing_y),
                                            resampleAlg=dem_resample_method,
                                            outputBounds=output_extent)

        dst_ds = gdal.Warp("", input_raster, options=gdalwarp_options)
        resampled_dem = dst_ds.ReadAsArray()

        # Save to the harddrive if the out_path is not None
        if out_path is not None:
            driver = gdal.GetDriverByName('GTiff')
            data_type = gdal.GDT_Float32
            des = driver.Create(out_path, out_width, out_length, 1, data_type)

            # Write data to the hard drive
            band = des.GetRasterBand(1)
            band.WriteArray(resampled_dem)

            # Set the geotransform
            geotransform = (out_start_x, out_spacing_x, 0,
                            out_start_y, 0, out_spacing_y)
            des.SetGeoTransform(geotransform)

            # Set the projection
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(out_epsg)
            des.SetProjection(srs.ExportToWkt())

    return resampled_dem


def interpolate_L2_datacube(nisar_L2_product_path,
                            dem_path,
                            cube_ds_name,
                            out_ds_path,
                            cube_interp_method = 'linear',
                            gunw_geogrid_ds_name = None,
                            frequency = 'A',
                            polarization = 'HH',
                            force_resample_dem = False,
                            dem_resample_method = 'cubicspline',
                            out_dem_file = None,
                            out_format = "GTiff"):
    """
    Interpolate the 3D metadata datacube in the NISAR L2 product
    """

    with h5py.File(nisar_L2_product_path) as f:
        # get the product type in the HDF5, and it must be
        # 'GCOV', 'GSLC', 'GUNW', or 'GOFF'
        product_type = f['science/LSAR/identification/productType'][()].astype(str)
        if product_type not in ['GCOV', 'GSLC', 'GUNW', 'GOFF']:
            err_str = f'Product type {product_type} is not supproted'
            raise ValueError(err_str)

        # Check the cube dataset name in the radar grid metadata cube
        if product_type == 'GCOV':
            if cube_ds_name not in GCOV_RADAR_GRID_DATASETS:
                err_str = f"{product_type} does not have {cube_ds_name} in its datacube,"
                " please pick up one of:\n"+\
                ','.join(GCOV_RADAR_GRID_DATASETS)
                raise ValueError(err_str)
        elif product_type == 'GSLC':
            if cube_ds_name not in GSLC_RADAR_GRID_DATASETS:
                err_str = f"{product_type} does not have {cube_ds_name} in its datacube,"
                " please pick up one of:\n"+\
                ','.join(GSLC_RADAR_GRID_DATASETS)
                raise ValueError(err_str)
        elif product_type == 'GOFF':
            if cube_ds_name not in GOFF_RADAR_GRID_DATASETS:
                err_str = f"{product_type} does not have {cube_ds_name} in its datacube,"
                " please pick up one of:\n"+\
                ','.join(GOFF_RADAR_GRID_DATASETS)
                raise ValueError(err_str)
        elif product_type == 'GUNW':
            if cube_ds_name not in GUNW_RADAR_GRID_DATASETS:
                err_str = f"{product_type} does not have {cube_ds_name} in its datacube,"
                " please pick up one of:\n"+\
                ','.join(GUNW_RADAR_GRID_DATASETS)
                raise ValueError(err_str)

        # Get the X, Y, and Z coordindates of the datacube
        xcoords = f[f'/science/LSAR/{product_type}'
                    '/metadata/radarGrid/xCoordinates'][()]
        ycoords = f[f'/science/LSAR/{product_type}'
                    '/metadata/radarGrid/yCoordinates'][()]
        zcoords = f[f'/science/LSAR/{product_type}'
                    '/metadata/radarGrid/heightAboveEllipsoid'][()]

        # Get the datacube datataset
        ds_name = f'/science/LSAR/{product_type}/metadata/radarGrid/{cube_ds_name}'
        if ds_name not in f:
            err_str = f'{ds_name} is not in the radar grid metadata cube of {product_type}'
            raise ValueError(err_str)

        # cube dataset
        ds_cube = f[ds_name][()]

        # Build the regular grid interpolator
        # and check the first dimension since the Baseline top/bottom mode only has 2 heights
        ds_cube_shape = ds_cube.shape
        if ds_cube_shape[0] == 2:
            interplator = RegularGridInterpolator(
                (np.array([zcoords[0], zcoords[-1]]),
                 ycoords, xcoords), ds_cube, method = cube_interp_method)
        else:
            interplator = RegularGridInterpolator(
                (zcoords, ycoords, xcoords),ds_cube, method = cube_interp_method)

        # Get the geogrid of the product in the L2 product and check if
        # the frequency is in the product
        group_name = f'/science/LSAR/{product_type}/grids/frequency{frequency}/'
        if group_name not in f:
            err_str = f'Frequency {frequency} is not in the product'
            raise ValueError(err_str)

        out_epsg = 4326
        if product_type == 'GUNW':
            if gunw_geogrid_ds_name is None:
                gunw_ds_name = 'unwrappedInterferogram'
            else:
                gunw_ds_name = gunw_geogrid_ds_name
            ds_x = f[f'{group_name}/{gunw_ds_name}/{polarization}/xCoordinates'][()]
            ds_y = f[f'{group_name}/{gunw_ds_name}/{polarization}/yCoordinates'][()]
            out_epsg = int(f[f'{group_name}/{gunw_ds_name}/{polarization}/projection'][()])
        elif product_type == 'GOFF':
            ds_x = f[f'{group_name}/pixelOffsets/{polarization}/layer1/xCoordinates'][()]
            ds_y = f[f'{group_name}/pixelOffsets/{polarization}/layer1/yCoordinates'][()]
            out_epsg = int(f[f'{group_name}/pixelOffsets/{polarization}/layer1/projection'][()])
        else:
            ds_x = f[f'{group_name}/xCoordinates'][()]
            ds_y = f[f'{group_name}/yCoordinates'][()]
            out_epsg = int(f[f'{group_name}/projection'][()])

        # Resample the DEM to be the same geogrid with the dataset
        ds_dem_data = resample_dem(dem_path, out_epsg,
                                   ds_x[0], ds_y[0],
                                   ds_x[1] - ds_x[0],
                                   ds_y[1] - ds_y[0],
                                   len(ds_y), len(ds_x),
                                   force_resample_dem,
                                   dem_resample_method,
                                   out_dem_file)

        # Build the meshgrid for the interpolator
        y, x = np.meshgrid(ds_y, ds_x,
                           indexing='ij')
        pts = np.stack([ds_dem_data.ravel(),
                        y.ravel(),
                        x.ravel()]).T

        # Interpolating the points
        interp_pts = interplator(pts)
        group_name = \
            f"NETCDF:{nisar_L2_product_path}://science/LSAR/{product_type}/grids/frequency{frequency}"
        if product_type == 'GCOV':
            src = rasterio.open(f"{group_name}/{[polarization]*2}",
                                driver = 'netCDF')
        if product_type == 'GSLC':
            src = rasterio.open(f"{group_name}/{polarization}",
                                driver = 'netCDF')
        if product_type == 'GUNW':
            if ((gunw_geogrid_ds_name is None) or
                (gunw_geogrid_ds_name == 'unwrappedInterferogram')):
                gunw_ds_name = f'{gunw_geogrid_ds_name}/{polarization}/coherenceMagnitude'
            elif gunw_geogrid_ds_name == 'wrappedInterferogram':
                gunw_ds_name = f'{gunw_geogrid_ds_name}/{polarization}/coherenceMagnitude'
            elif gunw_geogrid_ds_name == 'pixelOffsets':
                gunw_ds_name = f'{gunw_geogrid_ds_name}/{polarization}/alongTrackOffset'
            src = rasterio.open(f"{group_name}/{gunw_ds_name}",
                                driver = 'netCDF')
        if product_type == 'GOFF':
            src = rasterio.open(
                f"{group_name}/pixelOffsets/{polarization}/layer1/alongTrackOffset",
                driver = 'netCDF')

        # Write the data to the harddrive
        out_meta = src.meta.copy()
        out_meta.update({"driver": out_format,
                        "height": src.shape[0],
                        "width": src.shape[1],
                        "dtype": 'float32',
                        'nodata': 0.0,
                        })

        with rasterio.open(out_ds_path, "w", **out_meta) as dest:
            dest.write(interp_pts.reshape(src.shape),1)

def main():
    parser = get_parser()
    t_all = time.time()
    interpolate_L2_datacube(parser.input_file,
                            parser.input_dem_file,
                            parser.cube_dataset_name,
                            parser.output_file,
                            parser.cube_interp_method,
                            parser.gunw_dataset_name,
                            parser.frequency,
                            parser.polarization,
                            parser.force_resample_dem,
                            parser.dem_resample_method,
                            parser.out_resampled_dem,
                            parser.out_format)
    t_all_elapsed = time.time() - t_all
    print(f"Successfully ran the {parser.cube_dataset_name}"
          f" interpolation in {t_all_elapsed:.3f} seconds")

if __name__ == "__main__":
    main()