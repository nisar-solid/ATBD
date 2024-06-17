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

   '''
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
   '''
   # init dataframe
   df = pd.DataFrame(np.vstack([pair_distance,
                                pair_difference]).T,
                                columns=['distance', 'double_diff'])

   # remove nans
   df_nonan = df.dropna(subset=['double_diff'])
   bins = np.linspace(*distance_rqmt, num=n_bins+1)
   bin_centers = (bins[:-1] + bins[1:]) / 2
   binned_df = df_nonan.groupby(pd.cut(df_nonan['distance'], bins),
                                observed=False)[['double_diff']]

   # get binned validation table 
   validation = pd.DataFrame([])
   validation['total_count[#]'] = binned_df.apply(lambda x: np.ma.masked_invalid(x).count())
   validation['passed_req.[#]'] = binned_df.apply(lambda x: np.count_nonzero(x < requirement))
   
   # Add total at the end
   validation = pd.concat([validation, pd.DataFrame(validation.sum(axis=0)).T])
   validation['passed_pc'] = validation['passed_req.[#]'] / validation['total_count[#]']
   validation['success_fail'] = validation['passed_pc'] > threshold
   validation.index.name = 'distance'
   # Rename last row
   validation.rename({validation.iloc[-1].name:'Total'}, inplace=True)

   # Figure
   fig, ax = plt.subplots(1, figsize=(9, 3), layout="none", dpi=200)
   
   # Plot residuals
   ax.scatter(df_nonan.distance, df_nonan.double_diff,
              color='black', s=8, zorder=1, alpha=0.6, edgecolor='None')

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
            color='None', edgecolor=color, linewidth=2, zorder=2)
      
   # Add legend with data info
   legend_kwargs = dict(transform=ax.transAxes, verticalalignment='top')
   props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth=0.4)
   textstr = f'Sensor: {sensor} \n{validation_data}-InSAR pointpairs\n'
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
   fig.suptitle(f"{validation_type.capitalize()} requirement: {site_name}")
   ax.set_xlabel("Distance (km)")
   ax.set_ylabel("Double-Differenced \nVelocity Residual (mm/yr)")
   ax.minorticks_on()
   ax.tick_params(axis='x', which='minor', length=4, direction='in', top=False, width=1.5)
   ax.set_xticks(bin_centers, minor=True)
   ax.set_xticks(np.arange(0,55,5))
   ax.set_ylim(0,20)
   ax.set_xlim(*distance_rqmt)

   validation = validation.rename(columns={'success_fail': f'passed_req [>{threshold*100:.1f}%]'})

   return validation, fig