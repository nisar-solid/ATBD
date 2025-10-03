# Author: Marin Govorcin
# June, 2024
# Transient validation display function added by Saoussen Belhadj-aissa. July, 2024

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
                       validation_data:str='Field_meas'):

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
    distance_rqmt_log = [0.011,np.log10(distance_rqmt[1])]

    bins = np.logspace(*distance_rqmt_log, num=n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                observed=False)[['double_diff']]

    # Get binned validation table 
    validation = pd.DataFrame([])
    validation['total_count[#]'] = binned_df.apply(lambda x: np.ma.masked_invalid(x).count())
    validation['passed_req.[#]'] = binned_df.apply(lambda x: np.count_nonzero(x < requirement))
    bin_counts = validation['total_count[#]'].values
   
    # Add total at the end
    validation = pd.concat([validation, pd.DataFrame(validation.sum(axis=0)).T])
    validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
    validation['success_fail'] = validation['passed_pc'] > threshold
    validation.index.name = 'distance[km]'

    # Rename last row
    validation.rename({validation.iloc[-1].name:'Total'}, inplace=True)

    # Figure
    fig, ax = plt.subplots(1, figsize=(9, 3), layout="none", dpi=200)
    ax.set_xscale('log')
    
    # Plot residuals
    distance = np.log(df_nonan.distance)
    ms = 6 if pair_difference.shape[0] < 1e4 else 0.3
    alpha = 0.6 if pair_difference.shape[0] < 1e4 else 0.2
    ax.scatter(df_nonan.distance, df_nonan.double_diff,
               color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')

    ax.fill_between(distance_rqmt, 0, requirement, color='#e6ffe6', zorder=0, alpha=0.6)
    ax.fill_between(distance_rqmt, requirement, 31, color='#ffe6e6', zorder=0, alpha=0.6)
    ax.vlines(bins, 0, 31, linewidth=0.3, color='gray', zorder=1)
    ax.axhline(requirement, color='k', linestyle='--', zorder=3)

    # Bar plot for each bin
    quantile_th = binned_df.quantile(q=threshold)['double_diff'].values
    for bin_center, quantile, flag, bindiff in zip(bin_centers,
                                          quantile_th,
                                          validation['success_fail'],np.diff(bins)):

       if flag:
          color = '#227522'
       else:
          color = '#7c1b1b'

       ax.bar(bin_center, quantile, align='center', width=bindiff,
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

    # Plot number of points in bin
    ax2 = ax.twinx()
    ax2.plot(bin_centers, bin_counts, marker='o', linestyle='-')
    ax2.set_yscale('log')
    ax2.set_ylim(1+np.min(bin_counts), 10*np.max(bin_counts))
    
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
    ax.set_xticks([1,10,100])
    ax.set_ylim(0,30)
    ax.set_xlim([1,100])

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


def display_transient_validation(pair_distances: list, pair_differences: list, ifgs_dates: list,
                                 site_name: str, distance_rqmt: list = [0.1, 50],
                                 n_bins: int = 10, threshold: float = 0.683, sensor: str = 'Sentinel-1',
                                 validation_data: str = 'GNSS'):
    
    
    """
    Parameters:
      pair_distances : array     - lis of  1d array of pair distances used in validation
      pair_differences : array   - list of 1d array 0f pair double differenced displacement residuals
      site_name : str            - name of the cal/val site
      start_date  : str          - data record start date, eg. 20190101
      end_date : str             - data record end date, eg. 20200101
      distance_rqmt : list       - distance over requirement is tested, eg. length scales of 0.1-50 km
      n_bins : int               - number of bins
      threshold : float          - threshold represents percentile of Gaussian normal distribution
                                    within residuals are expected to be to pass the test
                                    e.g. 0.683 for 68.3% or 1-sigma limit 
      sensor : str               - sensor used in validation, e.g Sentinel-1 or NISAR
      validation_data : str      - data used to validate against; GNSS or INSAR

   Return
      validation_table : styled_df
      validation_figure : fig
    """
    validation_type = 'Transient'
    maxY=80 ## Y limit in the subplot
    
    n_ifgs = len(pair_distances) ## Number of interferograms to validate
    
    
    # Data frame initialization
    bins = np.linspace(*distance_rqmt, num=n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    columns = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}' for i in range(n_bins)] + ['Total']
    index = [f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}" for start, end in ifgs_dates]
    n_all = np.zeros([n_ifgs, n_bins + 1], dtype=int)
    n_pass = np.zeros([n_ifgs, n_bins + 1], dtype=int)
    
    ## Requirements per interferogram and each interferogram per bin
    for i in range(n_ifgs):
        inds = np.digitize(pair_distances[i], bins)
        for j in range(1, n_bins + 1):
            mask = inds == j
            rem = np.abs(pair_differences[i][mask])
            rqmt = 3 * (1 + np.sqrt(bins[j - 1]))
            n_all[i, j - 1] = len(rem)
            n_pass[i, j - 1] = np.sum(rem < rqmt)
        n_all[i, -1] = np.sum(n_all[i, :-1])
        n_pass[i, -1] = np.sum(n_pass[i, :-1])

    # Calculation of ratios and success/failure
    ratio = n_pass / np.where(n_all > 0, n_all, 1)
    success_or_fail = ratio > threshold
    
    # Creation of DataFrames for validation table
    n_all_pd = pd.DataFrame(n_all, columns=columns, index=index)
    n_pass_pd = pd.DataFrame(n_pass, columns=columns, index=index)
    ratio_pd = pd.DataFrame(ratio, columns=columns, index=index)
    success_or_fail_str = pd.DataFrame(success_or_fail.astype(str), columns=columns, index=index)

    ## Styling the DataFrame
    def style_specific_cells(val):
        color = '#e6ffe6' if val > threshold else '#ffe6e6'
        return f'background-color: {color}'

    # Apply style to all cells, and bold for 'Total' column
    styled_df = (ratio_pd.style.applymap(style_specific_cells)
                 .apply(lambda x: ['font-weight: bold' if x.name == 'Total' else '' for _ in x], axis=0))

    
    # Start subplot, each subplot represent validation test per interferogram 
    num_cols = 3 ## Can be changed to adjust subplot figure
    num_rows = (n_ifgs + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4*num_rows))
    
    axs = np.array(axs).reshape(num_rows, num_cols)

    for i in range(0,n_ifgs):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        ## Data frame for validation of each interferogram
        df = pd.DataFrame(np.vstack([pair_distances[i],
                                     pair_differences [i]]).T,
                                     columns=['distance', 'double_diff']) 
        # remove nans, draw bins and group by distance
        df_nonan = df.dropna(subset=['double_diff'])
        bins = np.linspace(*distance_rqmt, num=n_bins+1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                     observed=False)[['double_diff']]
        validation = pd.DataFrame(data={
            'total_count[#]': n_all[i],
            'passed_req.[#]': n_pass[i],
        })
        validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
        validation['success_fail'] = validation['passed_pc'] > threshold
        validation.index.name = 'distance[km]'
        validation.rename({validation.iloc[-1].name: 'Total'}, inplace=True)
        # start scatter plot 
        ms = 8 if len(pair_differences[i]) < 1e4 else 0.3
        alpha = 0.6 if len(pair_differences[i]) < 1e4 else 0.2

        ax.scatter(pair_distances[i], pair_differences[i],
                   color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')

        # Plot validation requirement log fit 
        dist_th = np.linspace(min(pair_distances[i]), max(pair_distances[i]), 100)
        acpt_error = 3 * (1 + np.sqrt(dist_th))
        ax.plot(dist_th, acpt_error, 'r')

        # Vertical lines for bins
        ax.vlines(bins, 0, maxY, linewidth=0.3, color='gray', zorder=1)

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
        textstr +=   f"Record: {index[i]}"

        # Place a text box in upper left in axes coords
        ax.text(0.02, 0.95, textstr, fontsize=7, bbox=props, **legend_kwargs)

        # Add legend with validation info
        textstr = f'{validation_type.capitalize()} Req \n'
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

        # Add colored rectangle indicating validation status
        rect = patches.Rectangle((0.8, 0.75), 0.19, 0.2,
                                 linewidth=1, edgecolor='black',
                                 facecolor=validation_color,
                                 transform=ax.transAxes)
        ax.add_patch(rect)

        # Title & labels
        ax.set_xlabel("Distance (km)", fontsize=8)
        if validation_data == 'GNSS':
            txt = "Double-Differenced \n Displacement Residual (mm)"
        else:
            txt = "Relative Velocity measurement (mm/yr)"    
        ax.set_ylabel(txt, fontsize=8)
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xticks(bin_centers, minor=True)
        ax.set_xticks(np.arange(0, 55, 5))
        ax.set_ylim(0, maxY)
        ax.set_xlim(*distance_rqmt)
        ax.set_title(f"Residuals \n Date range {index[i]} \n Number of station pairs used: {len(pair_distances[i])} \n Cal/Val Site Los Angeles " )
        
    # Hide unused subplots if there are any
    for idx in range(n_ifgs, num_rows*num_cols):
        axs.flat[idx].axis('off')  

    # Figure title
    fig.suptitle(f"{validation_type.capitalize()} requirement for site : {site_name} \n", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    ## In case we want to save the figure
    # plt.savefig(f'transient_validation_{index[i]}.png', bbox_inches='tight', transparent=True)
    
    plt.close()
    return styled_df, fig 

def display_permafrost_validation(pair_distance: NDArray, pair_difference: NDArray,
                       site_name: str, start_date: str, end_date: str,
                       requirement: float = 2, req_dist_fcn: bool = False,
                       distance_rqmt: list = [0.1, 50], n_bins: int = 10, threshold: float = 0.8, 
                       sensor:str ='Sentinel-1', validation_type:str='permafrost',
                       validation_data:str='field'):
   '''
    Parameters:
      pair_distance : array      - 1d array of pair distances used in validation
      pair_difference : array    - 1d array 0f pair double differenced velocity residuals
      site_name : str            - name of the cal/val site
      start_date  : str          - data record start date, eg. 20190101
      end_date : str             - data record end date, eg. 20200101
      requirement : float        - value required for test to pass
                                    e.g, 2 mm/yr for 3 years of data over distance requiremeent
      req_dist_fcn: bool         - flag to scale requirement with distance. If True, requirement = value * (1+sqrt(L))
      distance_rqmt : list       - distance over requirement is tested, eg. length scales of 0.1-50 km
      n_bins : int               - number of bins
      threshold : float          - threshold represents percentile of Gaussian normal distribution
                                    within residuals are expected to be to pass the test
                                    e.g. 0.8 for 80%
      sensor : str               - sensor used in validation, e.g Sentinel-1 or NISAR
      validation_type : str      - type of validation: permafrost
      validation_data : str      - data used to validate against; field or INSAR

   Return
      validation_table
      validation_figure
   '''
   # init dataframe
   pair_req_met = np.array(pair_difference < requirement*(1+float(req_dist_fcn)*np.sqrt(pair_distance)))
   pair_req_met[np.isnan(pair_difference)]=np.nan
   df = pd.DataFrame(np.vstack([pair_distance,
                                pair_difference,
                                pair_req_met]).T,
                                columns=['distance', 'double_diff','req_met'])

   # remove nans
   df_nonan = df.dropna(subset=['double_diff'])
   bins = np.linspace(*distance_rqmt, num=n_bins+1)
   bin_centers = (bins[:-1] + bins[1:]) / 2
   binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                observed=False)[['req_met']]
   binned_df_diff = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                observed=False)[['double_diff']]
    
   # get binned validation table 
   bin_req = requirement*(1+float(req_dist_fcn)*np.sqrt(bin_centers))
   validation = pd.DataFrame([])
   validation['total_count[#]'] = binned_df.apply(lambda x: np.ma.masked_invalid(x).count())
   validation['passed_req.[#]'] = binned_df.apply(lambda x: np.count_nonzero(x))
    
   # Add total at the end
   validation = pd.concat([validation, pd.DataFrame(validation.sum(axis=0)).T])
   validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
   validation['success_fail'] = validation['passed_pc'] > threshold
   validation.index.name = 'distance[km]'
   # Rename last row
   validation.rename({validation.iloc[-1].name:'Total'}, inplace=True)

   # Figure
   fig, ax = plt.subplots(1, figsize=(9, 3), layout="none", dpi=200)
   ymax = 20
   if validation_type=='permafrost':
       ymax=50
    
    
   # Plot residuals
   ms = 8 if pair_difference.shape[0] < 1e4 else 0.3
   alpha = 0.6 if pair_difference.shape[0] < 1e4 else 0.2
   ax.scatter(df_nonan.distance, df_nonan.double_diff,
              color='black', s=ms, zorder=1, alpha=alpha, edgecolor='None')

   for i,r in enumerate(bin_req):
       ibin = [bins[i],bins[i+1]]
       ax.fill_between(ibin, 0, r, color='#e6ffe6', zorder=0, alpha=0.6)
       ax.fill_between(ibin, r, 51, color='#ffe6e6', zorder=0, alpha=0.6)
   ax.vlines(bins, 0, ymax+1, linewidth=0.3, color='gray', zorder=1)

   req_line_x = np.linspace(*distance_rqmt, num=200)
   req_line_y = requirement*(1+req_dist_fcn*np.sqrt(req_line_x))
   ax.plot(req_line_x,req_line_y, color='k', linestyle='--', zorder=3)

   # Bar plot for each bin
   quantile_th = binned_df_diff.quantile(q=threshold)['double_diff'].values
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

   # place a text box in upper left in axes coords
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
       txt = "Relative displacement measurement (mm)"    
   ax.set_ylabel(txt, fontsize=8)
   ax.minorticks_on()
   ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
   ax.tick_params(axis='both', labelsize=8)
   ax.set_xticks(bin_centers, minor=True)
   ax.set_xticks(np.arange(0,55,5))
   ax.set_ylim(0,ymax)
   ax.set_xlim(*distance_rqmt)

   validation = validation.rename(columns={'success_fail': f'passed_req [>{threshold*100:.1f}%]'})

   return validation, fig

