# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:05:26 2021

@author: ErTodd
"""
# Deaggregation plotting subroutines
def sph2cart(r, theta, phi):
    '''spherical to cartesian transformation.'''
    import numpy as np
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    import numpy as np
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def ravzip(*itr):
    '''flatten and zip arrays'''
    import numpy as np
    return zip(*map(np.ravel, itr))

def create_epsilon_plot(df, plot_parameters):
    '''
    Creates DataFrame that reshapes deaggregation results into a form ready for plotting
    based on the epsilon bin and colour inputs.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame with deaggreation results in form columns=['mag','dist','eps','poe'].
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.

    Returns
    -------
    df_epsilon_plot : DataFrame
        Pandas DataFrame with deaggregation results ready for plotting. 
        columns=['mag','dist','epsilon bin (inclusive)','epsilon rgba','probability of exceedence']
    unique_mags : LIST
        List of magnitude bin centrepoints.
    unique_dists : LIST
        List of distance bin centrepoints.
    total_summed_poe : FLOAT
        Total sum of probability of exceedence values from deaggregation output.
        Used as base to calculate % hazard contributions by source.

    '''
    import numpy as np
    import pandas as pd
    
    epsilon_colours = plot_parameters['epsilon_colour_alpha_dictionary']
    unique_mags = np.unique(df.mag.to_numpy())
    unique_dists = np.unique(df.dist.to_numpy())
    
    total_summed_poe = sum([float(x) for x in df.poe.tolist()])


    df_epsilon_plot = pd.DataFrame(columns = ['mag','dist','eps_lower_bound','colour','poe'])

    if plot_parameters['trim_plot']:
        unique_mags = unique_mags[unique_mags < max(plot_parameters['ylim'])]
        unique_dists = unique_dists[unique_dists < max(plot_parameters['xlim'])]
        
    masterlist=[]
    for i, (lower_bound, color) in enumerate(epsilon_colours.items()):
        if i==0:
            eps_bin_interval = list(epsilon_colours.keys())[i+1] - lower_bound
        for m in unique_mags:
            for d in unique_dists:
                dftmp = df.loc[(df['mag']== m)
                               & (df['dist']==d)
                               & (df['eps']>=lower_bound)
                               & (df['eps']<(lower_bound + eps_bin_interval))]
                if len(dftmp)>1:
                    new_row_list = [m,d,lower_bound,color, float(dftmp['poe'].sum())]
                elif len(dftmp) == 1:
                    new_row_list = [m,d,lower_bound,color,float(dftmp['poe'].tolist()[0])]

                masterlist.append(new_row_list)


    df_epsilon_plot['mag'] = [x[0] for x in masterlist]
    df_epsilon_plot['dist'] = [x[1] for x in masterlist]
    df_epsilon_plot['eps_lower_bound'] = [x[2] for x in masterlist]
    df_epsilon_plot['colour'] = [x[3] for x in masterlist]
    df_epsilon_plot['poe'] = [x[4] for x in masterlist]
    df_epsilon_plot.sort_values(by=['mag','dist'],inplace=True,ignore_index=True)
    
    return df_epsilon_plot, unique_mags, unique_dists, total_summed_poe

def create_deag_Z_grid(ax, plot_parameters, unique_mags,unique_dists,df_epsilon_plot,this_sum):
    '''
    Create a three dimensional grid stacked with the % hazard contribution values for each epsilon bin
    NOTE: % hazard contribution stacks (z-axis) have negative epsilons at the bottom and positive epsilons at the top

    Parameters
    ----------
    ax : AXIS HANDLE
        Figure axis handle for modifying.
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    unique_mags : LIST
        List of magnitude bin centrepoints.
    unique_dists : LIST
        List of distance bin centrepoints.
    df_epsilon_plot : DataFrame
        Pandas DataFrame with deaggregation results ready for plotting. 
        columns=['mag','dist','epsilon bin (inclusive)','epsilon rgba','probability of exceedence']
    this_sum : FLOAT
        Total sum of probability of exceedence values intended for plotting. 
        May have been trimmed from original deaggregation output. Used to compute % total hazard values from poe

    Returns
    -------
    ax : AXIS HANDLE
        Modified figure axis handle.

    '''
    import numpy as np
    DZ_masterlist = []
    colour_masterlist = []
    
    dists_grid = []
    out = [dists_grid.append(unique_dists) for n in unique_mags]
    dist_grid_array = np.array(dists_grid)

    mags_grid = []
    out = [mags_grid.append(unique_mags) for n in unique_dists]
    mags_grid_array = np.array(mags_grid).T

    # make Z_grid for plotting
    Z_grid = np.zeros_like(dist_grid_array)

    count=0
    for mag_bar_i, m in enumerate(unique_mags):
        for dist_bar_i,d in enumerate(unique_dists):
            df_dz_tmp = df_epsilon_plot.loc[(df_epsilon_plot['mag']==m) & (df_epsilon_plot['dist']==d)]

            # Make DZ a percentage of the total hazard contribution of the plotted results
            # To do this, divide poe column by `this_sum` and multiply by 100 to plot in % form
            DZ_tmp = np.multiply(np.divide(df_dz_tmp['poe'].to_numpy(),this_sum),100)
            Z_grid[mag_bar_i,dist_bar_i] = count

            DZ_masterlist.append(DZ_tmp)
            colour_masterlist.append(df_dz_tmp['colour'].tolist())
            count=count+1
    
    ax = plot_deag_data(ax, plot_parameters, DZ_masterlist, colour_masterlist, 
                        Z_grid, dist_grid_array, mags_grid_array, 
                        unique_mags, unique_dists)

    return ax

def plot_deag_data(ax, plot_parameters, DZ_grid, Colour_grid, Z_grid, 
                   dist_grid_array, mags_grid_array, unique_mags, unique_dists):
    '''
    Plot deaggregation results with the following parameters: 
        - distance (x)
        - magnitude (y)
        - % hazard contribution (z)
        - epsilon (colour)

    Parameters
    ----------
    ax : AXIS HANDLE
        Figure axis handle for modifying.
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    DZ_grid : LIST
        List of magnitude bin lists containing a list of each % hazard contribution by epsilon bin for each distance.
        Format: [magnitude bins (y-axis) [distance bins (x-axis) [% hazard by epsilon bins (z-axis)]]]
    Colour_grid : LIST
        List of magnitude bin lists containing a list of the RGBA values (Red, Green, Blue, Alpha) 
        by epsilon bin for each distance.
        Format: [magnitude bins (y-axis) [distance bins (x-axis) [RGBA by epsilon bins (z-axis)]]]
    Z_grid : NUMPY ARRAY
        2-D numpy array of zeros (magnigude bins by distance bins) ready for filling by DZ_grid or Colour_grid.
    dist_grid_array : NUMPY ARRAY
        2-D numpy array with distance bins.
    mags_grid_array : NUMPY ARRAY
        2-D numpy array with magnitude bins.
    unique_mags : LIST
        List of magnitude bin centrepoints.
    unique_dists : LIST
        List of distance bin centrepoints.

    Returns
    -------
    ax : AXIS HANDLE
        Modified figure axis handle.

    '''
    import numpy as np

    #establish camera position
    xyz = np.array(sph2cart(*sphview(ax)), ndmin=3).T       #camera position in xyz
    zo = np.multiply([dist_grid_array, mags_grid_array, np.zeros_like(Z_grid)], xyz).sum(0)
    bars = np.empty(dist_grid_array.shape, dtype=object)
    #plot the data
    for i, (x,y,dz,o,cc) in enumerate(ravzip(dist_grid_array, mags_grid_array, Z_grid, zo, Z_grid)):
        _zpos = 0
        j, k = divmod(i,len(unique_dists))

        for ii,(dz,cz) in enumerate(zip(DZ_grid,Colour_grid)):
            if (Z_grid[j,k] == ii):
                for jj,(this_dz,this_colour) in enumerate(zip(dz,cz)):
                    # Only plot scenarios where dz[jj] >= plotting_tolerance
                    if dz[jj] >= plot_parameters['plotting_tolerance']:
                        bars[j,k] = pl = ax.bar3d(x, y, _zpos, plot_parameters['dx'], plot_parameters['dy'], dz[jj], 
                                                  color = cz[jj])    
        #wireframe option   bars[j,k] = pl = ax.bar3d(x, y, _zpos, dx, dy, dz[jj], color = (0,0,1,0), edgecolor = cz[jj], linewidth = 3)                              
                        pl._sort_zpos = o
                        _zpos += dz[jj]

    return ax    

def read_usgs18_deag_data(deagg_directory, plot_parameters):
    '''
    Reads deaggregation results in usgs18/opensha format and reorganises it into standard form for plotting.

    Parameters
    ----------
    deagg_directory : STR
        Path to directory containing data.csv results file for deaggregation.
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.

    Returns
    -------
    reshaped_df : DataFrame
        Pandas DataFrame containing deaggregation results reorganised with columns=['mag','dist','eps','poe'].

    '''
    import os
    import pandas as pd
    import numpy as np
    
    # Read in data.csv and assign trace (T) values to plot_parameters['trace_value']
    df = pd.read_csv(os.path.join(deagg_directory,'data.csv'), encoding='utf-8')
    
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace(to_replace = 'T', value = plot_parameters['trace_value'], inplace=True)
    
    #Rename columns to standardised headers (mostly unnecessary, but useful for looking at results)
    old_cols = df.columns.tolist()
    new_cols = [x.strip() for x in old_cols]
    df.rename(columns=dict(zip(old_cols,new_cols)),inplace=True)
    old_cols2 = ['r', 'r̅', 'm', 'm̅', 'ε̅', 'Σε']
    new_cols2 = ['dist','avg_dist','mag','avg_mag','avg_eps','summed_poe']
    df.rename(columns=dict(zip(old_cols2,new_cols2)),inplace=True)
    df.rename(columns=plot_parameters['epsilon_keys'],inplace=True)
    old_cols3 = list(plot_parameters['epsilon_keys'].values())
    new_cols3 = [x[1:-1].replace('-∞ ..','<').replace('.. +∞','<').replace('..','to') for x in old_cols3]
    new_cols4 = [x.split(' ')[0] if x.split(' ')[0] != '<' else '-3.0' for x in new_cols3]
    df.rename(columns=dict(zip(old_cols3,new_cols4)),inplace=True)
    
    # Get df into consistent format for plotting
    df1 = df[['mag','dist']]
    df2 = df[new_cols4]
    
    temp_df = pd.DataFrame(columns=['mag','dist','eps','poe'])
    
    reshaped_df = pd.DataFrame(columns=['mag','dist','eps','poe'])
    
    for index,row in df1.iterrows():
        temp_df['mag'] = np.tile(np.asarray(row[0]), len(new_cols4))
        temp_df['dist'] = np.tile(np.asarray(row[1]), len(new_cols4))
        temp_df['eps'] = [float(x) for x in new_cols4]
        temp_df['poe'] = df2.loc[index,:].tolist()
        
        reshaped_df = pd.concat([reshaped_df,temp_df])
    
    return reshaped_df

def parse_usgs18_deag_summary(deagg_directory):
    '''
    Read and parse the summary.txt file to get mean, mode, and epsilon bins

    Parameters
    ----------
    deagg_directory : STR
        Path to directory containing summary.txt file for deaggregation.

    Returns
    -------
    summary_dict : DICT
        Contents of summary.txt in dictionary form.

    '''
    import os
    
    with open(os.path.join(deagg_directory,'summary.txt'), encoding='utf8') as f:
        summary_data = f.readlines()
    #print(summary_data)
    keys = ['Deaggregation targets','Recovered targets','Totals','Mean (over all sources)','Mode (largest m-r bin)','Mode (largest m-r-ε₀ bin)', 'Discretization','Epsilon keys','·······']
    summary_dict = dict.fromkeys(keys[:-1])
    summary_dict_indices = dict.fromkeys(keys[:-1])

    for i,k in enumerate(keys[:-1]):
        this_section = []
        for linenum in range(2,len(summary_data)):
            if summary_data[linenum].startswith(k):
                this_section.append(int(linenum+1))
            elif summary_data[linenum].startswith(keys[i+1]):
                this_section.append(int(linenum))
        summary_dict_indices[k]=this_section

    #print(summary_dict_indices)  
    for key,indices in summary_dict_indices.items():
        nested_d_components = []
        for line in summary_data[indices[0]:indices[1]]:
            line_tuple = tuple(line.strip().split(':  '))
            if line_tuple == ('',):
                break
            else:
                nested_d_components.append(line_tuple)
        summary_dict[key]=dict(nested_d_components)

    return summary_dict

def set_zlim_zticks(plot_parameters):
    '''
    Compute zlim and zticks based on maximum hazard contribution across all IMT/return period pairs.

    Parameters
    ----------
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.

    Returns
    -------
    zlim : TUPLE
        Tuple denoting z limits of plot from 0 to the max hazard contribution.
    zticks : numpy array
        Numpy array with the start, stop, and step for labeling ticks on the z-axis.

    '''
    import numpy as np
    zlim = (0,plot_parameters['max_contribution'])
    if zlim[1] > 10:
        tick_space = 5
    elif zlim[1] > 3:
        tick_space = 2
    elif zlim[1] > 1:
        tick_space = 0.5
    else:
        tick_space = 0.25
            
    zticks = np.arange(0,plot_parameters['max_contribution'],tick_space)
    
    return zlim, zticks

def write_combined_deag_results_openquake(deag_results_filename_dict, 
                                          poe_OQlabel_dict,deagg_results_dir):
    '''
    Write deagg results to file for each IMT/return period pair from openquake rlz outputs

    Parameters
    ----------
    deag_results_filename_dict : DICT
        Dictionary with lists of deag result filepaths for each rlz file.
        Dictionary should be in the form: 
            deag_results_filename_dict[IMT][return_period] = [rlz_result1_filepath, rlz_result2_filepath, etc.]
    poe_OQlabel_dict : DICT
        Dictionary with the decimal return period (keys) and corresponding openquake file labels (values).
        Example: {'0.1' : 'poe-0', '0.04877' : 'poe-1', '0.02469' : 'poe-2', '0.00995' : 'poe-3', '0.00499' : 'poe-4'}
    deagg_results_dir : STR
         Directory containing OpenQuake deaggregation result files.
    '''
    import os
    import pandas as pd
    
    for sp in deag_results_filename_dict.keys():
        for poe,rlz_filenames in deag_results_filename_dict[sp].items():
            for i,rlz_filename in enumerate(rlz_filenames):
                dftmp = pd.read_csv(rlz_filename, skiprows=[0,1], 
                                    skip_blank_lines=True, engine='python')
                if i==0:
                    summed_poes = dftmp['poe'].to_numpy()
                else:
                    summed_poes += dftmp['poe'].to_numpy()
                    
            df=pd.DataFrame(columns=['mag','dist','eps','poe'])
            df['mag'] = dftmp['mag']
            df['dist'] = dftmp['dist']
            df['eps'] = dftmp['eps']
            df['poe'] = summed_poes
        
            df.to_csv(os.path.join(deagg_results_dir, 
                                   sp + '-sid-0-'+poe_OQlabel_dict[poe] 
                                   +'_Mag_Dist_Eps_'
                                   + rlz_filename.split('_')[-1]),
                      index=False)
            

# Deaggregation table subroutines
def initialise_deag_result_table(table_return_period_labels, deag_IMTs):
    '''
    Initialise the deaggregation mead, median, mode results table

    Parameters
    ----------
    table_return_period_labels : DICT
        Dictionary with return periods (in decimal form) and labels to use for the table.
        Format: {'0.1' : 'OBE - 1/475', '0.02469': 'SEE - 1/2,000', '0.00499' : '1/10,000'}
    deag_IMTs : LIST
        List of IMTs for the table.

    Returns
    -------
    report_df : DataFrame
        Pandas DataFrame with the column and row headers for the table. Initialised with zeros.

    '''
    import pandas as pd
    
    report_table_dictionary = dict.fromkeys(table_return_period_labels.keys())
    for return_period in report_table_dictionary.keys():   
        report_table_dictionary[return_period] = dict.fromkeys(deag_IMTs)
        for IMT in deag_IMTs:
            report_table_dictionary[return_period][IMT] = [[0],[0],[0],[0],[0],[0],[0],[0],[0]]
    
    reform = {(table_return_period_labels[outerKey],innerKey) : value
              for outerKey, innerDict in report_table_dictionary.items()
              for innerKey, value in innerDict.items()}

    report_df = pd.DataFrame(reform).T
    report_df.index.names = ['Performance Level - AEP','Spectral Period (s)']
    report_df.columns = pd.MultiIndex.from_tuples([('Distance (km)', 'Mean'),
                                                   ('Distance (km)', 'Median'),
                                                   ('Distance (km)', 'Mode'),
                                                   ('Magnitude (Mw)', 'Mean'),
                                                   ('Magnitude (Mw)', 'Median'),
                                                   ('Magnitude (Mw)', 'Mode'),
                                                   ('Epsilon', 'Mean'),
                                                   ('Epsilon', 'Median'),
                                                   ('Epsilon', 'Mode')])
    return report_df

def calculate_deag_mean_median_mode(deag_results_filepath_dict, mag_bin_size,dist_bin_size,epsilon_basis):
    '''
    Calculate the mean, median, and mode for Magnitude, Distance, and Epsilon 
    directly from the deaggregation output files

    Parameters
    ----------
    deag_results_filename_dict : DICT
        Dictionary with lists of deag result filepaths for each IMT/return period pair.
        Dictionary should be in the form: 
            deag_results_filename_dict[IMT][return_period] = filepath
    mag_bin_size : FLOAT
        Size of magnitude bins used for deaggregation.
    dist_bin_size : INT
        Size of distance bins used for deaggregation.
    epsilon_basis : STR
        The basis for the mean, median, mode calculations for Epsilon. Options: ['mean', 'median','mode'].

    Returns
    -------
    mean_median_mode : DICT
        Calculated mean, median, and mode for Magnitude, Distance, and Epsilon 
        for each IMT/return period pair.

    '''
    import sys
    import pandas as pd
    import numpy as np
    
    if not epsilon_basis in ['mean', 'median','mode']:
        print('Epsilon averages must be calculated based on Magnitude/Distance mean, median, or mode.              \nepsilon_basis must be one of ["mean", "median", "mode"]')
        sys.exit()
    
    # Initialise results dictionary
    mean_median_mode = dict.fromkeys(deag_results_filepath_dict.keys())
    
    for IMT in deag_results_filepath_dict.keys():
        mean_median_mode[IMT] = dict.fromkeys(deag_results_filepath_dict[IMT].keys())
        for return_period,filepath in deag_results_filepath_dict[IMT].items():
            
            mean_median_mode[IMT][return_period] = dict.fromkeys(['Magnitude', 'Distance', 'Epsilon'])
            for key in mean_median_mode[IMT][return_period].keys():
                mean_median_mode[IMT][return_period][key] = dict.fromkeys(['mode','mean','median'])
            
            df_avg = pd.read_csv(filepath)
            
            #Get Mag/Dist Mode
            df_avg_mode = df_avg.loc[df_avg['poe'] == max(df_avg['poe'].to_numpy())]
            
            mean_median_mode[IMT][return_period]['Magnitude']['mode'] = df_avg_mode['mag'].to_numpy()[0]
            mean_median_mode[IMT][return_period]['Distance']['mode'] = df_avg_mode['dist'].to_numpy()[0]
            
            # Get Mag/Dist Mean
            deag_results = pd.DataFrame(index= df_avg.dist.unique().tolist(), 
                                        columns = df_avg.mag.unique().tolist())
            for m in df_avg.mag.unique():
                for d in df_avg.dist.unique():    
                    tmp = df_avg.loc[(df_avg['mag']== m) & (df_avg['dist']==d)]
                    deag_results.at[d,m] = (tmp.poe.sum()/df_avg['poe'].sum())
            
            deag_results.loc['Mag_Sum',:]= deag_results.sum(axis=0)
            deag_results.loc[:,'Dist_Sum'] = deag_results.sum(axis=1)
                    
            mag_cumsum = np.cumsum(deag_results.loc['Mag_Sum'].to_numpy())
            dist_cumsum = np.cumsum(deag_results.loc[:,'Dist_Sum'].to_numpy())
            
            mag_weighted_mean = np.sum(np.multiply(deag_results.loc['Mag_Sum'].to_numpy()[:-1], df_avg.mag.unique()))
            mag_mean_bin = [x for x in df_avg.mag.unique()
                            if np.abs(np.subtract(mag_weighted_mean,x))<(mag_bin_size/2)][0]        
        
            dist_weighted_mean = np.sum(np.multiply(deag_results.loc[:,'Dist_Sum'].to_numpy()[:-1], df_avg.dist.unique()))
            dist_mean_bin = [y for y in df_avg.dist.unique() 
                             if np.abs(np.subtract(dist_weighted_mean,y))<(dist_bin_size/2)][0]
            
            mean_median_mode[IMT][return_period]['Magnitude']['mean'] = mag_mean_bin
            mean_median_mode[IMT][return_period]['Distance']['mean'] = dist_mean_bin
            
            # Get Mag/Dist Median
            mag_cumsum_df = pd.DataFrame({'mags': deag_results.columns,'cumsum':mag_cumsum})
            dist_cumsum_df = pd.DataFrame({'dist': deag_results.index,'cumsum':dist_cumsum})
            
            mag_cumsum_df_temp = mag_cumsum_df.loc[mag_cumsum_df['cumsum']>0.5].reset_index().head(1)
            dist_cumsum_df_temp = dist_cumsum_df.loc[dist_cumsum_df['cumsum']>0.5].reset_index().head(1)
            
            mean_median_mode[IMT][return_period]['Magnitude']['median'] = mag_cumsum_df_temp['mags'].to_numpy()[0]
            mean_median_mode[IMT][return_period]['Distance']['median'] = dist_cumsum_df_temp['dist'].to_numpy()[0]
            
            # Get Mean, Median, Mode of Epsilon for the epsilon_basis Mag/Dist
            df_epsilon = df_avg.loc[(df_avg['mag']==mean_median_mode[IMT][return_period]['Magnitude'][epsilon_basis]) & (df_avg['dist']==mean_median_mode[IMT][return_period]['Distance'][epsilon_basis])].reset_index(drop=True)
            sum_poe = np.sum(df_epsilon['poe'].to_numpy())
            
            #Mode
            max_poe = df_epsilon['poe'].max()
            mean_median_mode[IMT][return_period]['Epsilon']['mode'] = round(df_epsilon.loc[df_epsilon['poe'] == max_poe, 'eps'].iloc[0],1)
            
            #Median
            poe_cumsum = np.divide(np.cumsum(df_epsilon['poe'].to_numpy()),sum_poe)
            df_epsilon.loc[:,'poe_cumsum_percent_of_poe_sum'] = poe_cumsum
            median_epsilon_temp = df_epsilon.loc[df_epsilon['poe_cumsum_percent_of_poe_sum']>0.5]
            mean_median_mode[IMT][return_period]['Epsilon']['median'] = round(median_epsilon_temp['eps'].to_numpy()[0],1)
            
            #Mean
            df_epsilon.loc[:,'weighted_sum_percent_of_poe_sum'] = np.divide(np.multiply(df_epsilon['eps'].to_numpy(), df_epsilon['poe'].to_numpy()), sum_poe)
            mean_median_mode[IMT][return_period]['Epsilon']['mean'] = round(np.sum(np.divide(np.multiply(df_epsilon['eps'].to_numpy(),df_epsilon['poe'].to_numpy()),sum_poe)),1)
            
            
    return mean_median_mode

def populate_mean_median_mode_table(mean_median_mode, report_df, deag_results_filepath_dict, table_return_period_labels,mag_bin_size, dist_bin_size):
    '''
    Use the calculated mean, median, and mode values for Magnitude, Distance, and Epsilon 
    to populate a deaggregation results table report.

    Parameters
    ----------
    mean_median_mode : DICT
        Calculated mean, median, and mode for Magnitude, Distance, and Epsilon 
        for each IMT/return period pair.
    report_df : DataFrame
        Pandas DataFrame with the column and row headers for the table. Initialised with zeros.
    deag_results_filename_dict : DICT
        Dictionary with lists of deag result filepaths for each IMT/return period pair.
        Dictionary should be in the form: 
            deag_results_filename_dict[IMT][return_period] = filepath    
    table_return_period_labels : DICT
        Dictionary with return periods (in decimal form) and labels to use for the table.
        Format: {'0.1' : 'OBE - 1/475', '0.02469': 'SEE - 1/2,000', '0.00499' : '1/10,000'}
    mag_bin_size : FLOAT
        Size of magnitude bins used for deaggregation.
    dist_bin_size : INT
        Size of distance bins used for deaggregation.

    Returns
    -------
    report_df : TYPE
        DESCRIPTION.

    '''
    #print(deag_results_filepath_dict)
    for IMT in deag_results_filepath_dict.keys():
        for return_period,filepath in deag_results_filepath_dict[IMT].items():
            if return_period in list(table_return_period_labels.keys()):
                # Populate distance results
                report_df.loc[table_return_period_labels[return_period],IMT][('Distance (km)','Mean')] = str(int(mean_median_mode[IMT][return_period]['Distance']['mean'] - (dist_bin_size/2))) + ' - ' + str(int(mean_median_mode[IMT][return_period]['Distance']['mean'] + (dist_bin_size/2)))
                report_df.loc[table_return_period_labels[return_period],IMT][('Distance (km)','Median')] = str(int(mean_median_mode[IMT][return_period]['Distance']['median'] - (dist_bin_size/2))) + ' - ' + str(int(mean_median_mode[IMT][return_period]['Distance']['median'] + (dist_bin_size/2)))
                report_df.loc[table_return_period_labels[return_period],IMT][('Distance (km)','Mode')] = str(int(mean_median_mode[IMT][return_period]['Distance']['mode'] - (dist_bin_size/2))) + ' - ' + str(int(mean_median_mode[IMT][return_period]['Distance']['mode'] + (dist_bin_size/2)))
                
                # Populate magnitude results
                report_df.loc[table_return_period_labels[return_period],IMT][('Magnitude (Mw)','Mean')] = str(mean_median_mode[IMT][return_period]['Magnitude']['mean'] - (mag_bin_size/2)) + ' - ' + str(mean_median_mode[IMT][return_period]['Magnitude']['mean'] + (mag_bin_size/2))
                report_df.loc[table_return_period_labels[return_period],IMT][('Magnitude (Mw)','Median')] = str(mean_median_mode[IMT][return_period]['Magnitude']['median'] - (mag_bin_size/2)) + ' - ' + str(mean_median_mode[IMT][return_period]['Magnitude']['median'] + (mag_bin_size/2))
                report_df.loc[table_return_period_labels[return_period],IMT][('Magnitude (Mw)','Mode')] = str(mean_median_mode[IMT][return_period]['Magnitude']['mode'] - (mag_bin_size/2)) + ' - ' + str(mean_median_mode[IMT][return_period]['Magnitude']['mode'] + (mag_bin_size/2))
                
                # Populate Epsilon results
                report_df.loc[table_return_period_labels[return_period],IMT][('Epsilon','Mean')] = str(mean_median_mode[IMT][return_period]['Epsilon']['mean'])
                report_df.loc[table_return_period_labels[return_period],IMT][('Epsilon','Median')] = str(mean_median_mode[IMT][return_period]['Epsilon']['median'])
                report_df.loc[table_return_period_labels[return_period],IMT][('Epsilon','Mode')] = str(mean_median_mode[IMT][return_period]['Epsilon']['mode'])
                
    return report_df

def write_combined_deag_by_src_results_openquake(deag_results_filename_dict, source_models_deag, output_rlz_files, poe_OQlabel_dict, deag_resultsdir):
    
    import os
    import pandas as pd
    import numpy as np
    
    # From the output realizations file, identify the rlz IDs that correspond to each source model branch ID
    for output_rlz_file in output_rlz_files:
        output_rlz_df = pd.read_csv(output_rlz_file,skip_blank_lines=True)
        for sm_type in source_models_deag.keys():
            for sm,details2 in source_models_deag[sm_type].items():                
                df_tmp = output_rlz_df.loc[output_rlz_df['branch_path'].str.startswith(details2[0])]
                source_models_deag[sm_type][sm].append(df_tmp['rlz_id'].to_list())
    
    # Combine data from all realizations in each source model branch
    # and write a combined file to the deag results directory
    for sm_type in source_models_deag.keys():
        for sm_name,sm_details in source_models_deag[sm_type].items():
            
            for IMT in deag_results_filename_dict.keys():
                for return_period,rlz_filenames in deag_results_filename_dict[IMT].items():
                    #print('writing the file for',sm_type,sm_name,IMT,desired_poe,sm_details[0])
                    for i,thisrlz in enumerate(sm_details[2]):
                        #print('adding',thisrlz)
                        this_rlz_filename = [os.path.basename(x) for x in rlz_filenames
                                             if 'rlz-'+str(thisrlz) in x
                                             and IMT in x
                                             and poe_OQlabel_dict[return_period] in x][0]
                        df_tmp = pd.read_csv(os.path.join(deag_resultsdir,this_rlz_filename), 
                                             skiprows=[0,1],skip_blank_lines=True)
                        
                        #sum all rlz poes for this branch
                        if i==0:
                            summed_poes_thisbranch = df_tmp['poe'].to_numpy()
                        else:
                            summed_poes_thisbranch = np.add(df_tmp['poe'].to_numpy(),summed_poes_thisbranch)
                            
                    new_df = pd.DataFrame(columns=['mag','dist','eps','poe'])
                    new_df['mag']=df_tmp['mag']
                    new_df['dist'] = df_tmp['dist']
                    new_df['eps'] = df_tmp['eps']
                    new_df['poe'] = summed_poes_thisbranch
                    
                    deag_results_filename_dict[IMT][return_period][0].split(os.sep)[-1].split(IMT)[1]
                    new_df.to_csv(os.path.join(deag_resultsdir, sm_details[0]+'_'+IMT + '_'
                                               + deag_results_filename_dict[IMT][return_period][0].split(os.sep)[-1].split(IMT)[1]),index=False)