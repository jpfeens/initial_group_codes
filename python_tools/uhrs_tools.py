# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:44:25 2021

@author: ErTodd
"""

def read_uhrs_openquake(mean_rlz_or_fractile, filename_prefix, results_dir, OQrunnum):
    '''
    Find, read, and parse the uniform hazard response specrtum results from 
    OpenQuake for a given return period. OpenQuake fractile hazard output is 
    in a file called 'hazard_uhs-mean_OQrunnum.csv', 'hazard_uhs-rlz-XXX_OQrunnum.csv',
    or 'quantile_uhs-FRACTILE_OQrunnum.csv'.

    Parameters
    ----------
    mean_rlz_or_fractile : STR
        String containing 'mean', the rlz number (i.e. '005'), or the decimal 
        form of a fractile (i.e. '0.05' for 5th fractile).
    filename_prefix : STR
        String containing 'hazard_uhs', 'hazard_uhs-rlz', or 'quantile_uhs' 
        depending on the type of file to be parsed.
    results_dir : STR
         Directory containing OpenQuake mean hazard result files.
    OQrunnum : STR
        OpenQuake run identifier located in filename.

    Returns
    -------
    df : Pandas DataFrame
        DataFrame with UHRS acceleration values for each return period (column), IMT (index) pair.
    '''
    import os
    import sys
    import pandas as pd
    
    from misc_tools import openquake_header_checks
    
    filename = filename_prefix +'-' + mean_rlz_or_fractile + '_' + OQrunnum + '.csv'
    filepath = os.path.join(results_dir,filename)
    
    if not os.path.exists(filepath):
        print(filepath,'not found')
        sys.exit()
    
    # Read results while skipping blank lines
    with open(filepath) as f:
        lines = f.readlines()
    lines = [x for x in lines if not x=='\n']
    
    header_line = lines[0]
    aep_IMT_line = lines[1].strip()
    acceleration_line = lines[2].strip()
    
    # Check the header to ensure the correct results are read
    if mean_rlz_or_fractile == 'mean':
        kind='mean'
        openquake_header_checks(header_line,'uhrs',kind)
    elif mean_rlz_or_fractile.isdigit() == True:
        kind='rlz-'+ mean_rlz_or_fractile
        openquake_header_checks(header_line,'uhrs',kind)
    elif float(mean_rlz_or_fractile)<1:
        kind='quantile-' + mean_rlz_or_fractile
        openquake_header_checks(header_line,'uhrs',kind)
    else:
        print('mean_rlz_or_fractile must be a string containing "mean", the rlz number (i.e. "005"), or the decimal form of a fractile (i.e. "0.05" for 5th fractile).')
        sys.exit()
    
    # Parse the uhrs results into a dataframe 
    # Columns = return periods; Indices = IMTs
    columns = set([x.split('~')[0] for x in aep_IMT_line.split(',')[2:]])
    indices = set([x.split('~')[1] for x in aep_IMT_line.split(',')[2:]])
    
    df = pd.DataFrame(index = indices, columns = columns)
    for location,acceleration in zip(aep_IMT_line.split(',')[2:],acceleration_line.split(',')[2:]):
        col = location.split('~')[0]
        ind = location.split('~')[1]
        df.at[ind,col] = acceleration
        
    return df

