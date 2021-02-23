# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:30:52 2021

@author: ErTodd
"""

# OpenQuake Format

'''
Openquake Hazard Output format:
-------------------------------
#

lon       lat        depth    poe-0.0000700    poe-0.0001000...

137.09    -33.27389  0        9.88E-01         9.85E-01...
'''

def parse_openquake_hazard(IML_line,acceleration_line):
    '''
    Parse the hazard curve results in openquake format. Works for hazard curve and quantile curve results.

    '''
    import numpy as np
    
    IMLs = np.asarray([float(x.replace('poe-','')) for x in IML_line.split(',')[3:]])
    accelerations = np.asarray([float(x) for x in acceleration_line.split(',')[3:]])
    
    return IMLs, accelerations

def read_mean_hazard_openquake(IMT, results_dir, OQrunnum):
    '''
    Find, read, and parse the mean hazard results from OpenQuake for a given spectral period (IMT).
    OpenQuake mean hazard output is in a file called 'hazard_curve-mean-IMT-OQrunnum.csv'.
    
    Output format:
    --------------
        #
        
        lon       lat        depth    poe-0.0000700    poe-0.0001000...
        
        137.09    -33.27389  0        9.88E-01         9.85E-01...

    Parameters
    ----------
    IMT : STR
        Intensity Measure Type (or spectral period).
    results_dir : STR
        Directory containing OpenQuake mean hazard result files.
    OQrunnum : STR
        OpenQuake run identifier located in filename.

    Returns
    -------
    IMLs : numpy array
        Intensity measure level against which the mean hazard of the IMT is plotted. 
        Units are g (acceleration) for PGA and Spectral Acceleration IMTs and velocity for PGV.
    mean_accel : numpy array
        Mean hazard (acceleration) values for the given IMT.
    '''
    import os
    import sys
    
    from misc_tools import openquake_header_checks
    
    filename = 'hazard_curve-mean-' + IMT + '_' + OQrunnum + '.csv'
    filepath = os.path.join(results_dir,filename)
    
    if not os.path.exists(filepath):
        print(filepath,'not found')
        sys.exit()
    
    # Read results while skipping blank lines
    with open(filepath) as f:
        lines = f.readlines()
    lines = [x for x in lines if not x=='\n']
    
    header_line = lines[0]
    IML_line = lines[1]
    acceleration_line = lines[2]
    
    # Check the header to ensure the correct results are read
    kind='mean'
    openquake_header_checks(header_line,IMT,kind)
    
    # Parse the IML and acceleration values into numpy arrays
    IMLs, accelerations = parse_openquake_hazard(IML_line, acceleration_line)
    
    return IMLs, accelerations

def read_fractile_curve_openquake(fractile, IMT, results_dir, OQrunnum):
    '''
    Find, read, and parse the fractile hazard results from OpenQuake for a given spectral period (IMT) and FRACTILE.
    OpenQuake fractile hazard output is in a file called 'quantile_curve-FRACTILE-IMT_OQrunnum.csv'.

    Parameters
    ----------
    fractile : STR
        Fractile hazard curve in decimal form.
    IMT : STR
        Intensity Measure Type (or spectral period).
    results_dir : STR
        Directory containing OpenQuake mean hazard result files.
    OQrunnum : STR
        OpenQuake run identifier located in filename.

    Returns
    -------
    IMLs : numpy array
        Intensity measure level against which the mean hazard of the IMT is plotted. 
        Units are g (acceleration) for PGA and Spectral Acceleration IMTs and velocity for PGV.
    accelerations : numpy array
        Hazard (acceleration) values for the given fractile.

    '''
    import os
    import sys
    
    from misc_tools import openquake_header_checks
    
    filename = 'quantile_curve-' + fractile + '-' + IMT + '_' + OQrunnum + '.csv'
    filepath = os.path.join(results_dir,filename)
    
    if not os.path.exists(filepath):
        print(filepath,'not found')
        sys.exit()
    
    # Read results while skipping blank lines
    with open(filepath) as f:
        lines = f.readlines()
    lines = [x for x in lines if not x=='\n']
    
    header_line = lines[0]
    IML_line = lines[1]
    acceleration_line = lines[2]
    
    # Check the header to ensure the correct results are read
    kind='quantile-'+fractile
    openquake_header_checks(header_line,IMT,kind)
    
    # Parse the IML and acceleration values into numpy arrays
    IMLs, accelerations = parse_openquake_hazard(IML_line, acceleration_line)
    
    return IMLs, accelerations

def read_rlz_hazard_openquake(rlz, IMT, results_dir, OQrunnum):
    
    import os
    import sys
    
    from misc_tools import openquake_header_checks
    
    filename = 'hazard_curve-rlz-' + rlz + '-' + IMT + '_' + OQrunnum + '.csv'
    filepath = os.path.join(results_dir,filename)
    
    if not os.path.exists(filepath):
        print(filepath,'not found')
        sys.exit()
    
    # Read results while skipping blank lines
    with open(filepath) as f:
        lines = f.readlines()
    lines = [x for x in lines if not x=='\n']
    
    header_line = lines[0]
    IML_line = lines[1]
    acceleration_line = lines[2]
    
    # Check the header to ensure the correct results are read
    kind='rlz-'+rlz
    openquake_header_checks(header_line,IMT,kind)
    
    # Parse the IML and acceleration values into numpy arrays
    IMLs, accelerations = parse_openquake_hazard(IML_line, acceleration_line)
    
    return IMLs, accelerations

#def read_mean_hazard_usgs2018(IMT, results_dir):
    
    
#def read_mean_hazard_ezfrisk(IMT, results_dir):
    