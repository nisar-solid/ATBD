import os
from datetime import datetime as dt
import shutil

import json


def save_results(method:str, save_dir:str, site_info:dict,
                 validation_table, validation_fig):
    """
    I want to create a new folder with a timestamp and the method.
    In that folder, I want the diff-vs-dist validation fig,
    and text describing the input params that went into that,
    and the output table (dataframe) in text form.
    """
    # Create new directory for outputs
    if os.path.exists(save_dir):
        print(f"Directory {save_dir} exists")
    else:
        print(f"Creating {os.path.basename(save_dir)}")
        os.makedirs(save_dir)

    # Save input parameters
    params_fname = f"Method{method:s}_params.txt"
    print(f"Saving processing parameters to: {params_fname}")
    with open(os.path.join(save_dir, params_fname), 'w') as params_file:
        json.dump(site_info, params_file, indent=4)

    # Save validation table
    table_fname = f"Method{method:s}_validation_table.txt"
    print(f"Saving validation table to: {table_fname}")
    with open(os.path.join(save_dir, table_fname), 'w') as table_file:
        validation_table.to_csv(table_file)

    # Save validation figure
    fig_fname = f"Method{method:s}_validation_figure.png"
    print(f"Saving validation figure to: {fig_fname}")
    validation_fig.savefig(os.path.join(save_dir, fig_fname),
                           bbox_inches='tight', transparent=True, dpi=300)

    print(f"Saved parameters and results to: {save_dir:s}")

    # Save to zip file
    shutil.make_archive(save_dir, 'zip', save_dir)
