# Emre Havazli, Ekaterina Tymofyeyeva
# May 2025

import os
import subprocess
from mintpy.utils import readfile

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