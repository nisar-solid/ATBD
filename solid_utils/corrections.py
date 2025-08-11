# Emre Havazli, Ekaterina Tymofyeyeva, Robert Zinke
# May 2025

import os
import subprocess
import numpy as np
import h5py
from mintpy.utils import readfile, writefile, plot
from mintpy.objects import ifgramStack, timeseries

## Correction application
def run_cmd(command, desc=None, check=True):
    """Run a shell command with optional logging."""
    if desc:
        print("#" * 10, desc, "#" * 10)
    print(f"$ {command}")
    subprocess.run(command, shell=True, check=check)

def apply_cor(cor_name, ts_file, cor_file, config_file, mintpy_dir, output_ts, output_vel):
    """Apply correction to a MintPy time series file and estimate velocity."""
    
    # === Validate inputs ===
    if not os.path.isfile(ts_file):
        raise FileNotFoundError(f"Time series file not found: {ts_file}")
    else:
        print(f"Found time series file: {ts_file}")

    # === Load reference attributes from time series ===
    atr = readfile.read_attribute(ts_file)
    ref_lat = float(atr["REF_LAT"])
    ref_lon = float(atr["REF_LON"])
    ref_date = atr["REF_DATE"]

    # === Handle MintPy SET correction ===
    if cor_name.lower() == 'set':
        if not os.path.isfile(cor_file):
            run_cmd(f"solid_earth_tides.py {ts_file} -g {mintpy_dir}/inputs/geometryGeo.h5")

    # === Handle ionospheric correction ===
    elif cor_name.lower() == 'iono':
        run_cmd(f"modify_network.py {cor_file} -t {config_file}", desc="Modifying network for ionosphere")
        run_cmd(f"reference_point.py {cor_file} --lat {ref_lat} --lon {ref_lon}", desc="Setting reference point")
        run_cmd(f"ifgram_inversion.py {cor_file} --dset unwrapPhase --ref-date {ref_date} --weight-func no --update",
                desc="Estimating ionospheric delay time-series")

        cor_file = os.path.join(mintpy_dir, "ion.h5")


    run_cmd(f"reference_point.py {cor_file} --lat {ref_lat} --lon {ref_lon}", desc="Setting reference point")
    run_cmd(f"reference_date.py {cor_file} --ref-date {ref_date}", desc="Setting reference date")

    # === Apply correction ===
    run_cmd(f"diff.py {ts_file} {cor_file} -o {output_ts} --force", desc="Applying the correction")

    # === Estimate velocity ===
    run_cmd(f"timeseries2velocity.py {output_ts} -o {output_vel}", desc="Estimating velocity")

    print("#" * 10, "Correction and velocity estimation complete", "#" * 10)
    print("Time series:", os.path.abspath(output_ts))
    print("Velocity:   ", os.path.abspath(output_vel))

    return os.path.abspath(output_ts), os.path.abspath(output_vel)


## Interferogram stack correction
def pairwise_stack_from_timeseries(ifgs_file:str, tropo_cor_file:str):
    """Create a stack of differential troposphere correction layers from 
    tropo phase delay scenes that matches the sequence of interferograms 
    in the ifgs_file.

    MintPy convention notes:
    Ifgs are stored according to the ref_repeat (older_recent) convention.
    Ifgs are in units of radians, whereas correction layers are in units of 
    meters.

    Parameters:
        ifgs_file : str      - file containing interferograms 
                               (e.g., ifgramStack.h5)
        tropo_cor_file : str - file containing tropospheric correction layers
                               (e.g., HRRR_ARIA.h5)
    """
    # Read interferogram date pairs from ifgs_file
    ifg_datepairList = ifgramStack(ifgs_file).get_date12_list()
    
    # Read tropo corrections from tropo_cor_file
    tropo_dateList = timeseries(tropo_cor_file).get_date_list()

    # Ensure that each interferogram has corresponding correction layers
    ifg_dates = []
    for date_pair in ifg_datepairList:
        ifg_dates.extend(date_pair.split("_"))
    ifg_dates = list(set(ifg_dates))
    
    for date in ifg_dates:
        if date not in tropo_dateList:
            raise Exception(f"No tropo correction layer for {date}")

    # Read tropo metadata
    tropo_metadata = readfile.read_attribute(tropo_cor_file,
                                             datasetName="timeseries")

    # Scale to convert from meter to radians
    _, unit, scale = plot.scale_data2disp_unit(metadata=tropo_metadata,
                                               disp_unit="radian")

    # The scale is the negative of what is needed for the ref_repeat convention
    # presumably because MintPy flips the sign when computing the TS from the
    # ifgram stack, so the negative scale reverses the sign flip.
    # In our case, we are scaling the units while maintaining the same sign
    # convention, so we cancel out the sign flip.
    scale *= -1

    # Get array size of interferogram stack
    with h5py.File(ifgs_file, 'r') as ifgs:
        ifgs_shape = ifgs["unwrapPhase"].shape
    
    # Empty array for differential tropo layers
    tropo_diff_stack = np.empty(ifgs_shape)

    # Loop through interferogram pairs
    for i, datepair in enumerate(ifg_datepairList):
        # Parse dates - for MintPy convention, the more recent date is the reference
        ref_date, sec_date = datepair.split("_")
    
        # Retrieve tropo layers
        tropo_ref, _ = readfile.read(tropo_cor_file,
                                     datasetName=f"timeseries-{ref_date}")
        tropo_sec, _ = readfile.read(tropo_cor_file,
                                     datasetName=f"timeseries-{sec_date}")
    
        # Compute differential troposphere delay for given date pair
        tropo_diff = tropo_ref - tropo_sec

        # Convert m to radians
        tropo_diff *= scale

        # Write values to stack
        tropo_diff_stack[i,...] = tropo_diff

    # Format output name
    dirname, basename = os.path.split(tropo_cor_file)
    nameparts = basename.split(".")
    fext = nameparts.pop(-1)
    outname = f"{'.'.join(nameparts)}_stack.{fext}"
    outpath = os.path.join(dirname, outname)

    # Metadata
    metadata = readfile.read_attribute(tropo_cor_file)
    metadata['FILE_TYPE'] = "ifgramStack"

    num_pair = len(ifg_datepairList)
    length = int(metadata['LENGTH'])
    width = int(metadata['WIDTH'])
    ds_name_dict = {
        'date'             : (np.dtype('S8'), (num_pair, 2)),
        'dropIfgram'       : (np.bool_,       (num_pair,)),
        'bperp'            : (np.float32,     (num_pair,)),
        'unwrapPhase'      : (np.float32,     (num_pair, length, width)),
        "coherence"        : (np.float32,     (num_pair, length, width)),
    }

    # Save to file
    writefile.layout_hdf5(outpath, ds_name_dict=ds_name_dict, metadata=metadata)
    with h5py.File(outpath, 'a') as outfile:
        for i, datepair in enumerate(ifg_datepairList):
            d12 = datepair
            outfile['date'][i,0] = d12.split("_")[0].encode("utf-8")
            outfile['date'][i,1] = d12.split("_")[1].encode("utf-8")
            outfile['dropIfgram'][i] = True

            outfile['unwrapPhase'][i,...] = +1.0 * tropo_diff_stack[i,...]

    return outpath