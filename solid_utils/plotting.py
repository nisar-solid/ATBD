# Author: Marin Govorcin
# June, 2024
# Transient validation display function added by Saoussen Belhadjaissa. July, 2024

from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def display_validation(pair_distance: NDArray, pair_difference: NDArray,
                       site_name: str, start_date: str, end_date: str,
                       requirement: float = 2, distance_rqmt: list = [0.1, 50],
                       n_bins: int = 10, threshold: float = 0.683, 
                       sensor:str ='Sentinel-1', validation_type:str='secular',
                       validation_data:str='GNSS'):

    """Display double-difference validation results. Bin the double differences
    as a function of point separation, compute the stastics for each bin, and
    plot the results.

    Parameters:
       pair_distance : array      - 1d array of pair distances used in validation
       pair_difference : array    - 1d array 0f pair double differenced velocity residuals
       site_name : str            - name of the cal/val site
       start_date  : str          - data record start date, eg. 20190101
       end_date : str             - data record end date, eg. 20200101
       requirement : float        - value required for test to pass
                                     e.g, 2 mm/yr for 3 years of data over distance requiremeent
       distance_rqmt : list       - distance over requirement is tested, eg. length scales of 0.1-50 km
       n_bins : int               - number of bins
       threshold : float          - threshold represents percentile of Gaussian normal distribution
                                     within residuals are expected to be to pass the test
                                     e.g. 0.683 for 68.3% or 1-sigma limit 
       sensor : str               - sensor used in validation, e.g Sentinel-1 or NISAR
       validation_type : str      - type of validation: secular, coseismic, transient
       validation_data : str      - data used to validate against; GNSS or INSAR 

    Return
       validation_table
       validation_figure
    """
    # Init dataframe
    df = pd.DataFrame(np.vstack([pair_distance,
                                 pair_difference]).T,
                                 columns=['distance', 'double_diff'])

    # Remove nans
    df_nonan = df.dropna(subset=['double_diff'])
    bins = np.linspace(*distance_rqmt, num=n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                observed=False)[['double_diff']]

    # Get binned validation table 
    validation = pd.DataFrame([])
    validation['total_count[#]'] = binned_df.apply(lambda x: np.ma.masked_invalid(x).count())
    validation['passed_req.[#]'] = binned_df.apply(lambda x: np.count_nonzero(x < requirement))
   
    # Add total at the end
    validation = pd.concat([validation, pd.DataFrame(validation.sum(axis=0)).T])
    validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
    validation['success_fail'] = validation['passed_pc'] > threshold
    validation.index.name = 'distance[km]'

    # Rename last row
    validation.rename({validation.iloc[-1].name:'Total'}, inplace=True)

    # Figure
    fig, ax = plt.subplots(1, figsize=(9, 3), layout="none", dpi=200)

    # Plot residuals
    ms = 8 if pair_difference.shape[0] < 1e4 else 0.3
    alpha = 0.6 if pair_difference.shape[0] < 1e4 else 0.2
    ax.scatter(df_nonan.distance, df_nonan.double_diff,
               color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')

    ax.fill_between(distance_rqmt, 0, requirement, color='#e6ffe6', zorder=0, alpha=0.6)
    ax.fill_between(distance_rqmt, requirement, 21, color='#ffe6e6', zorder=0, alpha=0.6)
    ax.vlines(bins, 0, 21, linewidth=0.3, color='gray', zorder=1)
    ax.axhline(requirement, color='k', linestyle='--', zorder=3)

    # Bar plot for each bin
    quantile_th = binned_df.quantile(q=threshold)['double_diff'].values
    for bin_center, quantile, flag in zip(bin_centers,
                                          quantile_th,
                                          validation['success_fail']):
       if flag:
          color = '#227522'
       else:
          color = '#7c1b1b'
       ax.bar(bin_center, quantile, align='center', width=np.diff(bins)[0],
             color='None', edgecolor=color, linewidth=2, zorder=3)
      
    # Add legend with data info
    legend_kwargs = dict(transform=ax.transAxes, verticalalignment='top')
    props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0.4)
    textstr = f'Sensor: {sensor} \n{validation_data}-InSAR point pairs\n'
    textstr += f'Record: {start_date}-{end_date}'

    # Place a text box in upper left in axes coords
    ax.text(0.02, 0.95, textstr, fontsize=8, bbox=props, **legend_kwargs)

    # Add legend with validation info 
    textstr = f'{validation_type.capitalize()} requirement\n'
    textstr += f'Site: {site_name}\n'
    if validation.loc['Total']['success_fail']:
       validation_flag = 'PASSED'
       validation_color = '#239d23'
    else: 
       validation_flag ='FAILED'
       validation_color = '#bc2e2e'

    props = {**props, **{'facecolor':'none', 'edgecolor':'none'}}
    ax.text(0.818, 0.93, textstr, fontsize=8, bbox=props, **legend_kwargs)
    ax.text(0.852, 0.82,  f"{validation_flag}",
            fontsize=10, weight='bold',
            bbox=props, **legend_kwargs)

    rect = patches.Rectangle((0.8, 0.75), 0.19, 0.2,
                            linewidth=1, edgecolor='black',
                            facecolor=validation_color,
                            transform=ax.transAxes)
    ax.add_patch(rect)

    # Title & labels
    fig.suptitle(f"{validation_type.capitalize()} requirement: {site_name}", fontsize=10)
    ax.set_xlabel("Distance (km)", fontsize=8)
    if validation_data == 'GNSS':
        txt = "Double-Differenced \nVelocity Residual (mm/yr)"
    else:
        txt = "Relative Velocity measurement (mm/yr)"    
    ax.set_ylabel(txt, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticks(bin_centers, minor=True)
    ax.set_xticks(np.arange(0,55,5))
    ax.set_ylim(0,20)
    ax.set_xlim(*distance_rqmt)

    validation = validation.rename(columns={'success_fail': f'passed_req [>{threshold*100:.1f}%]'})

    return validation, fig

def display_validation_table(validation_table):
    # Display Statistics
    def bold_last_row(row):
        is_total = row.name == 'Total'
        styles = ['font-weight: bold; font-size: 14px; border-top: 3px solid black' if is_total else '' for _ in row]
        return styles
    
    def style_success_fail(value):
        color = '#e6ffe6' if value else '#ffe6e6'
        return 'background-color: %s' % color

    # Overall pass/fail criterion
    if validation_table.loc['Total'][validation_table.columns[-1]]:
        print("This velocity dataset passes the requirement.")
    else:
        print("This velocity dataset does not pass the requirement.")

    return (validation_table.style
            .bar(subset=['passed_pc'], vmin=0, vmax=1, color='gray')
            .format(lambda x: f'{x*100:.0f}%', na_rep="none", precision=1, subset=['passed_pc'])
            .apply(bold_last_row, axis=1)
            .map(style_success_fail, subset=[validation_table.columns[-1]])
           )


def display_coseismic_validation(pair_distance: NDArray, pair_difference: NDArray,
                                 site_name: str, start_date: str, end_date: str,
                                 requirement: Callable, distance_rqmt: list = [0.1, 50],
                                 n_bins: int = 10, threshold: float = 0.683,
                                 sensor:str ='Sentinel-1', validation_type:str='secular',
                                 validation_data:str='GNSS'):
    """Display double-difference validation results. Evaluate the pass/fail
    criterion for each double-difference measurement at a given distance. Then,
    bin the double differences as a function of point separation, compute the
    stastics for each bin, and plot the results.

    Parameters:
       pair_distance : array      - 1d array of pair distances used in validation
       pair_difference : array    - 1d array 0f pair double differenced velocity residuals
       site_name : str            - name of the cal/val site
       start_date  : str          - data record start date, eg. 20190101
       end_date : str             - data record end date, eg. 20200101
       requirement : lambda       - formula for validation requirement
       distance_rqmt : list       - distance over requirement is tested, eg. length scales of 0.1-50 km
       n_bins : int               - number of bins
       threshold : float          - threshold represents percentile of Gaussian normal distribution
                                    within residuals are expected to be to pass the test
                                    e.g. 0.683 for 68.3% or 1-sigma limit 
       sensor : str               - sensor used in validation, e.g Sentinel-1 or NISAR
       validation_type : str      - type of validation: secular, coseismic, transient
       validation_data : str      - data used to validate against; GNSS or INSAR

    Return
       validation_table
       validation_figure
    """
    # Init dataframe
    df = pd.DataFrame(np.vstack([pair_distance,
                                 pair_difference]).T,
                                 columns=['distance', 'double_diff'])

    # Apply requirement
    df['requirement'] = df['double_diff'] < requirement(df['distance'])
    df['requirement'] = df['requirement'].astype(int)

    # Remove nans
    df_nonan = df.dropna(subset=['double_diff'])

    # Bin data
    bins = np.linspace(*distance_rqmt, num=n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_series = pd.cut(df_nonan['distance'], bins)
    binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                 observed=False)[['double_diff', 'requirement']]
    
    # Get binned validation table
    validation = pd.DataFrame([])
    validation['total_count[#]'] = binned_df.apply(lambda x: np.ma.masked_invalid(x['double_diff']).count())
    validation['passed_req.[#]'] = binned_df.apply(lambda x: np.sum(x['requirement']))

    # Add total at the end
    validation = pd.concat([validation, pd.DataFrame(validation.sum(axis=0)).T])
    validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
    validation['success_fail'] = validation['passed_pc'] > threshold
    validation.index.name = 'distance[km]'

    # Rename last row
    validation.rename({validation.iloc[-1].name:'Total'}, inplace=True)

    # Figure
    fig, ax = plt.subplots(1, figsize=(9, 3), layout="none", dpi=200)

    # Plot residuals
    ms = 8 if pair_difference.shape[0] < 1e4 else 0.3
    alpha = 0.6 if pair_difference.shape[0] < 1e4 else 0.2
    ax.scatter(df_nonan.distance, df_nonan.double_diff,
               color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')

    distance_envelope = np.linspace(*distance_rqmt, num=100)
    requirement_envelope = requirement(distance_envelope)
    ax.fill_between(distance_envelope, 0, requirement_envelope, color='#e6ffe6', zorder=0, alpha=0.6)
    ax.fill_between(distance_envelope, requirement_envelope, 51, color='#ffe6e6', zorder=0, alpha=0.6)
    ax.vlines(bins, 0, 51, linewidth=0.3, color='gray', zorder=1)
    ax.plot(distance_envelope, requirement_envelope, color='k', linestyle='--', zorder=3)

    # Bar plot for each bin
    quantile_th = binned_df.quantile(q=threshold)['double_diff'].values
    for bin_center, quantile, flag in zip(bin_centers,
                                          quantile_th,
                                          validation['success_fail']):
        if flag:
            color = '#227522'
        else:
            color = '#7c1b1b'
        ax.bar(bin_center, quantile, align='center', width=np.diff(bins)[0],
               color='None', edgecolor=color, linewidth=2, zorder=3)

    # Add legend with data info
    legend_kwargs = dict(transform=ax.transAxes, verticalalignment='top')
    props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0.4)
    textstr = f"Sensor: {sensor} \n{validation_data}-InSAR point pairs\n"
    textstr += f"Record: {start_date}-{end_date}"

    # Place a text box in upper left in axes coords
    ax.text(0.02, 0.95, textstr, fontsize=8, bbox=props, **legend_kwargs)

    # Add legend with validation info
    textstr = f"{validation_type.capitalize()} requirement\n"
    textstr += f"Site: {site_name}\n"
    if validation.loc['Total']['success_fail']:
        validation_flag = 'PASSED'
        validation_color = '#239d23'
    else:
        validation_flag ='FAILED'
        validation_color = '#bc2e2e'

    props = {**props, **{'facecolor':'none', 'edgecolor':'none'}}
    ax.text(0.818, 0.93, textstr, fontsize=8, bbox=props, **legend_kwargs)
    ax.text(0.852, 0.82,  f"{validation_flag}",
            fontsize=10, weight='bold',
            bbox=props, **legend_kwargs)

    rect = patches.Rectangle((0.8, 0.75), 0.19, 0.2,
                             linewidth=1, edgecolor='black',
                             facecolor=validation_color,
                             transform=ax.transAxes)
    ax.add_patch(rect)

    # Title & labels
    fig.suptitle(f"{validation_type.capitalize()} requirement: {site_name}", fontsize=10)
    ax.set_xlabel("Distance (km)", fontsize=8)
    if validation_data == 'GNSS':
        txt = "Double-Differenced \nVelocity Residual (mm)"
    else:
        txt = "Relative Velocity measurement (mm)"
    ax.set_ylabel(txt, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticks(bin_centers, minor=True)
    ax.set_xticks(np.arange(0, 55, 5))
    ax.set_ylim(0, 50)
    ax.set_xlim(*distance_rqmt)

    validation = validation.rename(columns={'success_fail': f'passed_req [>{threshold*100:.1f}%]'})
    
    return validation, fig


##############################

def display_transient_validation(pair_distances: list, pair_differences: list, ifgs_dates: list,
                                 site_name: str, distance_rqmt: list = [0.1, 50], n_bins: int = 10,
                                 threshold: float = 0.683, sensor: str = 'Sentinel-1',
                                 validation_data: str = 'GNSS'):
    """
    Displays transient validation results for interferometric data.

    Parameters:
      pair_distances : list   - List of 1D arrays of pair distances used in validation.
      pair_differences : list - List of 1D arrays of pair double-differenced displacement residuals.
      ifgs_dates : list       - List of tuples containing interferogram start and end dates.
      site_name : str         - Name of the calibration/validation site.
      distance_rqmt : list    - Distance over which the requirement is tested (e.g., [0.1, 50] km).
      n_bins : int            - Number of bins for distance grouping.
      threshold : float       - Percentile of the Gaussian distribution within which residuals must fall (e.g., 0.683 = 1σ).
      sensor : str            - Sensor used in validation (e.g., Sentinel-1 or NISAR).
      validation_data : str   - Data source used for validation (e.g., GNSS or InSAR).

    Returns:
      styled_df (pandas Styler): Styled validation table.
      fig (matplotlib Figure): Validation plots.
    """
    validation_type = "Transient"
    max_y = 80  
    n_ifgs = len(pair_distances)

    # Define bins for distance grouping
    bins = np.linspace(*distance_rqmt, num=n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    columns = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(n_bins)] + ["Total"]
    index = [f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}" for start, end in ifgs_dates]

    # Initialize matrices
    n_all, n_pass = np.zeros((n_ifgs, n_bins + 1), dtype=int), np.zeros((n_ifgs, n_bins + 1), dtype=int)

    # Compute validation results n_pass: compare each gnss/insar residual with the threshold of transient requirements "rqmt = 3 * (1 + np.sqrt(bin_distances))"
    for i, (distances, differences) in enumerate(zip(pair_distances, pair_differences)):
        inds = np.digitize(distances, bins)

        for j in range(n_bins):
            mask = inds == j + 1
            bin_distances = distances[mask]
            rem = np.abs(differences[mask])
            rqmt = 3 * (1 + np.sqrt(bin_distances))
            n_all[i, j] = len(rem)
            n_pass[i, j] = np.sum(rem < rqmt)

        n_all[i, -1], n_pass[i, -1] = np.sum(n_all[i, :-1]), np.sum(n_pass[i, :-1])

    # Compute pass/fail ratios
    ratio = np.divide(n_pass, np.where(n_all > 0, n_all, 1))
    ratio_pd = pd.DataFrame(ratio, columns=columns, index=index)

    # Compute stack validation
    pass_percentage = np.count_nonzero(ratio_pd["Total"] > threshold) / n_ifgs
    stack_pass = pass_percentage >= threshold
    decision_message = "This Interferograms stack passes the transient validation" if stack_pass else "This Interferograms stack fails the transient validation"

    ratio_pd.loc["Conclusion"] = ["-"] * (len(columns) - 1) + [decision_message]

    # Styling functions
    def style_cells(val):
        if isinstance(val, (int, float)):
            return f'background-color: {"#e6ffe6" if val > threshold else "#ffe6e6"}'
        elif isinstance(val, str) and ("passes" in val or "fails" in val):
            return f'background-color: {"#e6ffe6" if "passes" in val else "#ffe6e6"}; font-weight: bold'
        return ""

    styled_df = (ratio_pd.style
                 .map(style_cells)
                 .format(lambda x: f"{x:.0%}" if isinstance(x, float) else x))

    # Plotting
    num_cols = 3
    num_rows = (n_ifgs + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axs = np.array(axs).reshape(num_rows, num_cols)

    for i, ax in enumerate(axs.flat):
        if i >= n_ifgs:
            ax.axis("off")
            continue

        df = pd.DataFrame({"distance": pair_distances[i], "double_diff": pair_differences[i]}).dropna()
        # binned_df = df.groupby(pd.cut(df["distance"], bins))["double_diff"]
        binned_df = df.groupby(pd.cut(df["distance"], bins), observed=False)["double_diff"]

        # Scatter
        ax.scatter(df.distance, df.double_diff, color="black", s=8, alpha=0.6, edgecolor="None")

        # Requirement curve
        dist_th = np.linspace(*distance_rqmt, 100)
        ax.plot(dist_th, 3 * (1 + np.sqrt(dist_th)), "r")

        # Bars
        quantile_values = binned_df.quantile(q=threshold).fillna(0)
        for bin_center, quantile, flag in zip(bin_centers, quantile_values, ratio[i, :-1] > threshold):
            if quantile == 0:
                continue
            color = "#227522" if flag else "#7c1b1b"
            ax.bar(bin_center, quantile, align="center", width=np.diff(bins)[0], color='None', edgecolor=color, linewidth=2, zorder=3)

        ax.set_xlabel("Distance (km)", fontsize=8)
        ax.set_ylabel("Double-Differenced Displacement Residual (mm)", fontsize=8)
        ax.set_xlim(*distance_rqmt)
        ax.set_ylim(0, max_y)
        ax.set_title(f"Residuals \n Date range {index[i]}\n {len(pair_distances[i])} station pairs")

    fig.suptitle(f"{validation_type.capitalize()} validation for site: {site_name}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.close()

    return styled_df, fig
