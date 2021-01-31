# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:12:02 2021

@author: ErTodd
"""

# Main Plotting Routines

def plot_mean_hazard(plot_parameters, results_dir, return_periods):
    '''
    Plot the mean hazard curves for the desired spectral periods.

    Parameters
    ----------
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    results_dir : STR
        File path to results files ready for plotting.
    return_periods : DICT
        Dictionary with selected return periods (AEP) and associated parameters for plotting.
    '''
    import os
    import sys
    import matplotlib.pyplot as plt
    from cycler import cycler

    from hazard_curve_tools import read_mean_hazard_openquake

    # Check that necessary parameters are present in plot_parameters
    required_parameters = ['output_format', 'IMTs_to_plot', 'IMT_labels', 
                           'colour_map', 'figsize', 'plotsave','savedir']
    if not all(parameter in plot_parameters for parameter in required_parameters):
        print('Some parameters are missing :', set(required_parameters) - plot_parameters.keys())
        sys.exit()
        

    # Set up colour cycler
    color_cycler = cycler('color', [plt.get_cmap(plot_parameters['colour_map'])(
        i/len(plot_parameters['IMTs_to_plot'])) for i in range(len(plot_parameters['IMTs_to_plot']))])
    plt.rc('axes', prop_cycle=color_cycler)

    # Set up figure
    fig, ax = plt.subplots(figsize=plot_parameters['figsize'])

    # Plot results for each desired spectral period
    for IMT, IMT_label in zip(plot_parameters['IMTs_to_plot'], plot_parameters['IMT_labels']):

        # Read data for plotting based on output type
        if plot_parameters['output_format'].lower() in ['openquake', 'open quake', 'oq']:
            #Check that openquake specific parameters are present in plot_parameters
            oq_parameters = ['OQrunnum']
            if not all(parameter in plot_parameters for parameter in oq_parameters):
                print('Some openquake parameters are missing : ', 
                      set(oq_parameters) - plot_parameters.keys())
                sys.exit()

            IMLs, accelerations = read_mean_hazard_openquake(IMT, results_dir, plot_parameters['OQrunnum'])
#        elif plot_parameters['output_format'].lower() in ['opensha', 'open sha', 'usgs2018', 'usgs 2018', 'usgs18', 'usgs 18']:
#            IMLs, accelerations = read_mean_hazard_usgs2018(IMT, results_dir)
#        elif plot_parameters['output_format'].lower() in ['ezfrisk', 'ez-frisk', 'ez frisk']:
#            IMLs, accelerations = read_mean_hazard_ezfrisk(IMT, results_dir)
        else:
            print('Data output type not known.')
            sys.exit()

        ax.loglog(IMLs, accelerations, label='%s' % IMT_label, linewidth=2)

    # Plot selected YRP
    ax = ax_plot_YRPs(ax, plot_parameters, return_periods)

    # Annotate and style plot
    ax = ax_plot_annotation_and_styling(ax, plot_parameters)

    # Save figure
    if plot_parameters['plotsave']:
        plt.savefig(os.path.join(plot_parameters['savedir'], 'mean_hazard_curve_' +
                                 '_'.join(plot_parameters['IMTs_to_plot']) + '.png'),
                    format='PNG', dpi=600, bbox_inches='tight', pad_inches=0.1
                    )

def plot_fractile_curves(plot_parameters, results_dir, fractile_hazard_curves, return_periods):
    '''
    Plot the hazard curve for each of the calculated qualtiles for a perscribed IMT (often the fundamental period)

    Parameters
    ----------
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    results_dir : STR
        File path to results files ready for plotting.
    fractile_hazard_curves : DICT
        Dictionary with selected fractile labels (keys) and associated parameters for plotting (values: [fractile, linestyle, color]).
    return_periods : DICT
        Dictionary with selected return periods (AEP) and associated parameters for plotting.
    '''    
    import os
    import sys
    import matplotlib.pyplot as plt

    from hazard_curve_tools import read_mean_hazard_openquake, read_fractile_curve_openquake
    
    # Set up figure
    fig, ax = plt.subplots(figsize=plot_parameters['figsize'])
    
    # Identify the IMT for plotting and check that there is only one
    IMT = plot_parameters['IMTs_to_plot']
    if isinstance(IMT, list) and len(IMT) == 1:
        IMT = IMT[0]
    elif isinstance(IMT,list) and len(IMT) > 1:
        print('Fractile plots are only made for a single IMT.\nPlease try again for a single spectral period.')
        sys.exit()
    
    # Plot results for each desired fractile
    for fractile_label, fractile_details in fractile_hazard_curves.items():
        fractile = fractile_details[0]

        # Read data for plotting based on output type
        if plot_parameters['output_format'].lower() in ['openquake', 'open quake', 'oq']:
            OQrunnum = plot_parameters['OQrunnum']
            if fractile_label.lower() == 'mean':
                IMLs, accelerations = read_mean_hazard_openquake(IMT, results_dir, OQrunnum)
            else:
                IMLs, accelerations = read_fractile_curve_openquake(fractile, IMT, results_dir, OQrunnum)
#        elif plot_parameters['output_format'].lower() in ['opensha', 'open sha', 'usgs2018', 'usgs 2018', 'usgs18', 'usgs 18']:
#            IMLs, mean_accel = read_mean_hazard_usgs2018(IMT, results_dir)
#        elif plot_parameters['output_format'].lower() in ['ezfrisk', 'ez-frisk', 'ez frisk']:
#            IMLs, mean_accel = read_mean_hazard_ezfrisk(IMT, results_dir)
        else:
            print('Data output type not known.')
            sys.exit()

        # Assign appropriate label for legend
        if fractile_label == 'mean':
            line_label = fractile_label
        else:
            line_label = r'%s$^{th}$ fractile' % fractile_label

        # Plot lines
        ax.loglog(IMLs, accelerations, label=line_label, linestyle = fractile_details[1], color=fractile_details[2], linewidth=2)

    # Plot selected YRP
    ax = ax_plot_YRPs(ax, plot_parameters, return_periods)

    # Annotate and style plot
    ax = ax_plot_annotation_and_styling(ax, plot_parameters)
    
    # Save figure
    if plot_parameters['plotsave']:
        plt.savefig(os.path.join(plot_parameters['savedir'], 'fractile_hazard_curves_' + IMT + '.png'),
                    format='PNG', dpi=600, bbox_inches='tight', pad_inches=0.1
                    )

def plot_mean_uhrs_spectra(plot_parameters,results_dir, return_periods):
    '''
    Plot mean uhrs for all return periods

    Parameters
    ----------
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    results_dir : STR
        File path to results files ready for plotting.
    return_periods : DICT
        Dictionary with selected return periods (AEP) and associated parameters for plotting.
    '''
    import os
    import matplotlib.pyplot as plt

    from uhrs_tools import read_uhrs_openquake
    
    # Set up figure
    fig, ax = plt.subplots(figsize=plot_parameters['figsize'])
    
    # Mean urhs
    mean_or_fractile = 'mean'
    filename_prefix = 'hazard_uhs'
    
    # Read data for plotting based on output type
    if plot_parameters['output_format'].lower() in ['openquake', 'open quake', 'oq']:
        OQrunnum = plot_parameters['OQrunnum']
        df = read_uhrs_openquake(mean_or_fractile, filename_prefix, results_dir, OQrunnum)
#    elif plot_parameters['output_format'].lower() in ['opensha', 'open sha', 'usgs2018', 'usgs 2018', 'usgs18', 'usgs 18']:
#        IMLs, mean_accel = read_uhrs_usgs2018(IMT, results_dir)
#    elif plot_parameters['output_format'].lower() in ['ezfrisk', 'ez-frisk', 'ez frisk']:
#        IMLs, mean_accel = read_uhrs_ezfrisk(IMT, results_dir)

    # Plot UHRS mean for each return period
    for yrp, yrp_details in return_periods.items():
        yrp_decimal = yrp_details[0]

        # Extract the IMTs and accelerations for the desired return period and replace PGA with provided proxy value
        IMTs = [float(x.replace('PGA',plot_parameters['pga_proxy'])) if x == 'PGA'
                else float(x.replace('SA(','').replace(')','')) for x in df.index.tolist()]
        accelerations = [float(x) for x in df[yrp_decimal].tolist()]
        
        # Sort the arrays by IMTs for plotting
        sorted_arrays = sorted(zip(IMTs, accelerations))
        IMTs, accelerations = [list(tuple) for tuple in  zip(*sorted_arrays)]

        # Plot lines
        ax.loglog(IMTs, accelerations, label='1:%s AEP' % yrp, color=yrp_details[1], linewidth=3)
    
    # Annotate and style plot
    ax = ax_plot_annotation_and_styling(ax, plot_parameters)
    
    # Save figure
    if plot_parameters['plotsave']:
        plt.savefig(os.path.join(plot_parameters['savedir'], 
                                 'mean_uhrs_' + '_'.join(return_periods.keys()) + '.png'),
                    format='PNG', dpi=600, bbox_inches='tight', pad_inches=0.1
                    )
    
def plot_uhrs_by_return_period(plot_parameters,results_dir, return_periods, fractile_hazard_curves):
    '''
    Plot mean and all fractile uhrs curves for each return period.

    Parameters
    ----------
     plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    results_dir : STR
        File path to results files ready for plotting.
    return_periods : DICT
        Dictionary with selected return periods (AEP) and associated parameters for plotting.
    '''
    import os
    import matplotlib.pyplot as plt

    from uhrs_tools import read_uhrs_openquake
    
    # Read data for plotting based on output type
    if plot_parameters['output_format'].lower() in ['openquake', 'open quake', 'oq']:
        #Initialise results dictionary to read and store results for each fractile
        results_dict = dict.fromkeys({v[0] for v in fractile_hazard_curves.values()})
        OQrunnum = plot_parameters['OQrunnum']
        
        for mean_or_fractile in results_dict.keys():
            # Set filename prefix based on mean_or_fractile
            if mean_or_fractile == 'mean':
                filename_prefix = 'hazard_uhs'
            else:
                filename_prefix = 'quantile_uhs'

            #Read the results and store in results_dict for each fractile
            results_dict[mean_or_fractile] = read_uhrs_openquake(mean_or_fractile, filename_prefix, results_dir, OQrunnum)
            
#    elif plot_parameters['output_format'].lower() in ['opensha', 'open sha', 'usgs2018', 'usgs 2018', 'usgs18', 'usgs 18']:
#        IMLs, mean_accel = read_uhrs_usgs2018(IMT, results_dir)
#    elif plot_parameters['output_format'].lower() in ['ezfrisk', 'ez-frisk', 'ez frisk']:
#        IMLs, mean_accel = read_uhrs_ezfrisk(IMT, results_dir)
    
    # Create a figure for each return period
    for yrp, yrp_details in return_periods.items():
        yrp_decimal = yrp_details[0]
        
        # Set up figure
        fig, ax = plt.subplots(figsize=plot_parameters['figsize'])
        
        # Extract IMTs and accelerations for each fractile and plot
        for fractile_label, fractile_details in fractile_hazard_curves.items():
            fractile = fractile_details[0]
            
            IMTs = [float(x.replace('PGA',plot_parameters['pga_proxy'])) if x == 'PGA'
                else float(x.replace('SA(','').replace(')','')) for x in results_dict[fractile].index.tolist()]
            accelerations = [float(x) for x in results_dict[fractile][yrp_decimal].tolist()]
            
            # Sort the arrays by IMTs for plotting
            sorted_arrays = sorted(zip(IMTs, accelerations))
            IMTs, accelerations = [list(tuple) for tuple in  zip(*sorted_arrays)]

            # Set curve labels
            if fractile == 'mean':
                line_label = '1:%s AEP mean' % yrp
                line_width = 3
            else:
                line_label = r'1:%s AEP %s$^{th}$ fractile' % (yrp, fractile_label)
                line_width = 2

            # Plot lines
            ax.loglog(IMTs, accelerations, label=line_label, color=yrp_details[1], 
                      linestyle=fractile_details[1], linewidth=line_width)
    
        # Annotate and style plot
        ax = ax_plot_annotation_and_styling(ax, plot_parameters)
        
        # Save figure
        if plot_parameters['plotsave']:
            plt.savefig(os.path.join(plot_parameters['savedir'], 'uhrs_' + yrp + '.png'),
                        format='PNG', dpi=600, bbox_inches='tight', pad_inches=0.1
                        )
    
        
def plot_deaggregation(deag_results_filepath_dict,plot_parameters):
    '''
    Main deaggregation plotting routine.

    Parameters
    ----------
    deag_results_filepath_dict : DICT
        Dictionary of filepaths containing results for each IMT, return period pair. 
        Dictionary should be in the form: 
            deag_results_filepath_dict[IMT][return_period] = results_filepath
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    '''
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    
    from deaggregation_tools import parse_usgs18_deag_summary, read_usgs18_deag_data, create_epsilon_plot, create_deag_Z_grid
    
    # Create a plot for each IMT/return period pair
    for IMT in deag_results_filepath_dict.keys():
        for return_period, results_dir in deag_results_filepath_dict[IMT].items():
            
            # Read data for plotting based on output type
            # OpenQuake format
            if plot_parameters['output_format'].lower() in ['openquake', 'open quake', 'oq']:
                OQrunnum = plot_parameters['OQrunnum']
                df = pd.read_csv(deag_results_filepath_dict[IMT][return_period])
                # Clean up variables for labels
                if IMT == 'PGA':
                    clean_IMT = 'PGA'
                else:
                    clean_IMT = IMT.replace('SA(','').replace(')','')
                return_period = [k for k,v in plot_parameters['yrps'].items() if v[0] == return_period]#[0]
                print(return_period)
                    
            # USGS2018 or OpenSHA format
            if plot_parameters['output_format'].lower() in ['opensha', 'open sha', 'usgs2018', 
                                                            'usgs 2018', 'usgs18', 'usgs 18']:
                summary_dict = parse_usgs18_deag_summary(results_dir)
                #print(summary_dict)
                max_deag_distance = float(summary_dict['Discretization']['r'].split(',')[1].strip().split('=')[1].strip())
                if max_deag_distance > max(plot_parameters['xlim']):
                    print(IMT, return_period,'\nWARNING: Deaggregation max distance',max_deag_distance, 
                          'km is larger than max plotting distance',max(plot_parameters['xlim']), 
                          'km.\nRecompute deaggregation as necessary.')
                plot_parameters.update({'epsilon_keys' : summary_dict['Epsilon keys']})
                df = read_usgs18_deag_data(results_dir, plot_parameters)
                # Clean up variables for labels
                if IMT == 'PGA':
                    clean_IMT = 'PGA'
                else:
                    clean_IMT = IMT.replace('SA','').replace('P','.')
                
        #    elif plot_parameters['output_format'].lower() in ['ezfrisk', 'ez-frisk', 'ez frisk']:
        #        IMLs, mean_accel = read_uhrs_ezfrisk(IMT, results_dir)
            
            #epsilon_colours = plot_parameters['epsilon_colour_alpha_dictionary']
        
            df_epsilon_plot, unique_mags, unique_dists, total_summed_poe = create_epsilon_plot(df, plot_parameters)
        
            if plot_parameters['trim_plot']:
                original_summed_poe = total_summed_poe
                total_summed_poe = sum([float(x) for x in df_epsilon_plot.poe.tolist()])
                if (1-(total_summed_poe/original_summed_poe)) > 0.03:
                    print('WARNING:', (1-(total_summed_poe/original_summed_poe))*100, 
                          '% of the hazard lies outside the plotted area.\nAdjust xlim and ylim as needed.\n')
        
            unique_mags.sort()
            unique_dists.sort()
            
            # set up the figure
            fig, ax = deag_fig_set_up(plot_parameters)
        
            # Create the plot
            ax = create_deag_Z_grid(ax, plot_parameters, unique_mags, unique_dists, df_epsilon_plot, 
                                    total_summed_poe)
        
            # annotate the plot
            ax.set_xlabel(plot_parameters['xlabel'], fontsize=plot_parameters['fontsize'], labelpad=15)
            ax.set_ylabel(plot_parameters['ylabel'], fontsize=plot_parameters['fontsize'], labelpad=15)
            ax.set_zlabel(plot_parameters['zlabel'], fontsize=plot_parameters['fontsize'], labelpad=15)
            legend_elements = build_deag_legend(plot_parameters['epsilon_colour_alpha_dictionary'])
            if IMT == 'PGA':
                legend_title = 'Deaggregation for 1:'+ return_period+' AEP at ' + clean_IMT
            else:
                legend_title = 'Deaggregation for 1:'+ return_period+' AEP at ' + clean_IMT + ' s'
            
            ax.legend(handles=legend_elements[::-1], ncol = plot_parameters['legend_ncol'], loc=plot_parameters['legend_loc'],
                      fontsize=plot_parameters['fontsize']-6, title= legend_title, 
                      title_fontsize=plot_parameters['fontsize']-2)
        
            if plot_parameters['plotsave']:
                plt.savefig(os.path.join(plot_parameters['savedir'],'deagg_' + IMT + '_' + return_period + '.png'), 
                            dpi=600,pad_inches=0.1,bbox_inches='tight')

def plot_hazard_by_gmm():
    
    print('Starting to write a function to plot hazard curves by GMM')


# General Plotting Routines

def ax_plot_annotation_and_styling(ax, plot_parameters):
    '''
    Annotate and style plots

    Parameters
    ----------
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.

    Returns
    -------
    ax : axes handle

    '''
    import sys
    #from itertools import compress
    
    # Check that necessary parameters are present in plot_parameters
    required_parameters = ['xlabel','ylabel','ftsize','axis_bounds','grid_on','x_ticks','x_tick_labels','y_ticks']
    if not all(parameter in plot_parameters for parameter in required_parameters):
        print('Some parameters are missing :', set(required_parameters) - plot_parameters.keys())
        sys.exit()
    optional_parameters = ['legend_title','legend_loc','legend_ftsize']
    if not all(parameter in plot_parameters for parameter in optional_parameters):
        print('Some optional parameters are missing and can be set: ', 
              set(optional_parameters) - plot_parameters.keys())
        
    # Axis Labels
    ax.set_xlabel(plot_parameters['xlabel'], weight='heavy',
               fontsize=plot_parameters['ftsize'], labelpad=2)
    ax.set_ylabel(plot_parameters['ylabel'], weight='heavy',
               fontsize=plot_parameters['ftsize'], labelpad=2)
    # Plot limits and grid
    ax.axis(plot_parameters['axis_bounds'])
    ax.grid(plot_parameters['grid_on'], which='both')
    
    # Legend
    
    # Still working out how to simplify the legend arguments based on keys passed in plot_parameters
    # Perhaps this is the place to create a class to handle all the argument permutations
    #legend_parameters = ['legend_title','legend_loc','legend_ftsize']
    #legend_parameters_bool = [parameter in plot_parameters for parameter in legend_parameters]
    #legend_parameters_present = list(compress(legend_parameters, legend_parameters_bool))
    
    if 'legend_title' in plot_parameters:
        ax.legend(edgecolor='inherit', borderpad=0.7, labelspacing=0.5, handlelength=2.5,
               fontsize=plot_parameters['ftsize']-6, title=plot_parameters['legend_title'])
    elif 'legend_loc' in plot_parameters:
        ax.legend(edgecolor='inherit', borderpad=0.7, labelspacing=0.5, handlelength=2.5,
               fontsize=plot_parameters['ftsize']-6, loc=plot_parameters['legend_loc'])
    elif 'legend_ftsize' in plot_parameters:
        ax.legend(edgecolor='inherit', borderpad=0.7, labelspacing=0.5, handlelength=2.5,
               fontsize=plot_parameters['legend_ftsize'])
    #elif 'legend_title' and 'legend_loc' in plot_parameters:
    #    ax.legend(edgecolor='inherit', borderpad=0.7, labelspacing=0.5, handlelength=2.5,
    #           fontsize=plot_parameters['ftsize']-6, title=plot_parameters['legend_title'], 
    #           loc=plot_parameters['legend_loc'])
    else:
        ax.legend(edgecolor='inherit', borderpad=0.7, labelspacing=0.5, handlelength=2.5,
               fontsize=plot_parameters['ftsize']-6)


    # Ticks and tick labels
    ax.tick_params(axis='both', which='major', pad=7)
    ax.set_xticks(plot_parameters['x_ticks'])
    ax.set_xticklabels(plot_parameters['x_tick_labels'], fontsize=plot_parameters['ftsize'])
    ax.set_yticks(plot_parameters['y_ticks'])
    # Format y-tick labels to print decimals rather than scientific notation
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:.1g}'.format(x) if x >= 1 else '{:.9f}'.format(x).rstrip('0') for x in vals],
                       fontsize=plot_parameters['ftsize'])
    
    return ax


def ax_plot_YRPs(ax, plot_parameters, return_periods):
    '''
    Plot horizontal lines and labels on hazard curve plots at desired return periods (e.g. OBE, SEE, MCE performance levels)

    Parameters
    ----------
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.
    return_periods : DICT
        Dictionary with selected return periods (AEP) and associated parameters for plotting.

    Returns
    -------
    ax : axes handle

    '''

    '''plot horizontal lines on hazard curve plots at desired AEPs (e.g. OBE, SEE, MCE performance levels)'''
    import sys
    # Check that the necessary parameters are in plot_parameters
    required_parameters = ['axis_bounds','yrp_label_position','ftsize']
    if not all(parameter in plot_parameters for parameter in required_parameters):
        print('Some parameters are missing :', set(required_parameters) - plot_parameters.keys())
        sys.exit()

    for yrp, yrp_details in return_periods.items():
        yrp = int(yrp)
        yrp_label = yrp_details[2]
        ax.hlines(1/yrp, plot_parameters['axis_bounds'][0]/10,
                   plot_parameters['axis_bounds'][1]*10, colors='k', linestyles='solid', linewidth=2)
        ax.annotate(yrp_label,  # this is the text
                     # this is the point to label
                     (plot_parameters['yrp_label_position'], 1/yrp),
                     textcoords="offset points",  # how to position the text
                     xytext=(0, -20),  # distance from text to points (x,y)
                     ha='center',  # horizontal alignment can be left, right or center
                     fontsize=plot_parameters['ftsize'])
    return ax

def deag_fig_set_up(plot_parameters):
    '''
    Set up the deaggregation plot with parameters from plot_parameters

    Parameters
    ----------
    plot_parameters : DICT
        Dictionary with parameters for plot including the spectral periods to plot and plot aesthetics.

    Returns
    -------
    fig : figure handle

    ax : axis handle

    '''
    import matplotlib.pyplot as plt
    import sys
    # Check that the necessary parameters are in plot_parameters
    required_parameters = ['figsize','xlim','ylim','zlim','xticks','yticks','zticks','ftsize']
    if not all(parameter in plot_parameters for parameter in required_parameters):
        print('Some parameters are missing :', set(required_parameters) - plot_parameters.keys())
        sys.exit()
    
    # Set up the figure
    fig = plt.figure(figsize=plot_parameters['figsize'])
    ax = plt.axes(projection = '3d')
    ax.view_init(ax.elev, ax.azim)
    
    ax.set_xlim(plot_parameters['xlim'])
    ax.set_ylim(plot_parameters['ylim'])
    ax.set_zlim(plot_parameters['zlim'])

    ax.set_xticks(plot_parameters['xticks'])
    ax.set_yticks(plot_parameters['yticks'])
    ax.set_zticks(plot_parameters['zticks'])

    ax.set_xticklabels(plot_parameters['xticks'], fontsize = plot_parameters['ftsize'])
    ax.set_yticklabels(plot_parameters['yticks'], fontsize = plot_parameters['ftsize'])
    ax.set_zticklabels(plot_parameters['zticks'], fontsize = plot_parameters['ftsize'])

    ax.view_init(ax.elev, ax.azim+10)
    
    return fig, ax

def build_deag_legend(data):
    '''
    Build a legend for matplotlib plt from dictionary of 
    epsilon lower bounds (keys) and corresponding RGBA (values).

    Parameters
    ----------
    data : DICT
        Dictionary containing lower bin values for epsilon and corresponding RGBA parameters.

    Returns
    -------
    legend_elements : legend handle

    '''
    from matplotlib.lines import Line2D
    
    legend_elements = []
    eps_bin_size = list(data.keys())[1] - list(data.keys())[0]
    for i, (key,value) in enumerate(data.items()):
        eps_range_label = str(key)+'≥ ε <'+str(key+eps_bin_size)
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                      label=eps_range_label, 
                                      markerfacecolor=value, markersize=16))
    return legend_elements
