# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:26:24 2021

@author: ErTodd
"""

hcm = 'hazard_curve-mean'
hcr = 'hazard_curve-rlz'

def read_hcurve_rlz_outcsv(output_filetype, filenames, investigation_time=50, ver='3.10'):
    import os
    import numpy as np
    
    if len(filenames) == 0:
        print("No filenames were provided.")
    elif type(filenames) is not list:
        print("Please provide a list of filenames.")
    else:
        print('moving on')

    
        #initialise masterlists
    accel_master_list=[]
    aep_master_list=[]
    poe_master_list=[]
    rlz_list = []

    for filename in filenames:
        
        # extract rlz from the header for hcr files and check that the header matches the filename
        # Extract the rlz
        mrlz = str(os.path.basename(filename).split('-')[2])
        #print('mrlz= ', mrlz)
        with open(filename) as f:
            lines = f.readlines()
        if ver == '3.10':
            lines = [lines[i] for i,x in enumerate(lines) if not x == '\n']

        header = lines[0]
        rlz = header.split('=')[4].split(',')[0][-4:-1]
        if rlz == mrlz:
            acceleration = []
            poe = []

            # Read the acceleration values 
            accel_tmp = lines[1].split(',')[3:]
            for a in accel_tmp:
                a=a[4:]
                acceleration.append(a)

            # Read the poe values
            poe = lines[2].split(',')[3:]

            # Clean the poe and accel data
            poe[-1] = poe[-1][:-1]   
            acceleration[-1] = acceleration[-1][:-1]

            # Create numpy arrays of the data
            acceleration = np.asarray(acceleration,dtype=float)
            poe = np.asarray(poe,dtype=float)

            # Convert probability of exceedence to annual excedence probabilty
            aep = np.divide(np.multiply(-1,np.log(np.subtract(1,poe))),investigation_time)

            # Append to master lists
            rlz_list.append(rlz)
            accel_master_list.append(acceleration)
            aep_master_list.append(aep)
            poe_master_list.append(poe)

    return rlz_list, accel_master_list, aep_master_list, poe_master_list

def read_mean_quantile_outcsv(output_filetype, filenames, investigation_time=50,ver='3.10'):

    import numpy as np
    
    print(output_filetype, filenames, investigation_time)

    if len(filenames) == 0:
        print("No filenames were provided.")
    elif type(filenames) is not list:
        print("Please provide a list of filenames.")

    accel_master_list=[]
    aep_master_list=[]
    poe_master_list=[]

    for filename in filenames:
        #print(filename)
        with open(filename) as f:
            lines = f.readlines()
        if ver == '3.10':
            lines = [lines[i] for i,x in enumerate(lines) if not x == '\n']

        acceleration = []
        poe = []

        # Read the acceleration values 
        accel_tmp = lines[1].split(',')[3:]
        for a in accel_tmp:
            a=a[4:]
            acceleration.append(a)

        #print(lines,'\nacceleration',acceleration)

        # Read the poe values
        poe = lines[2].split(',')[3:]

        # Clean the poe and accel data
        poe[-1] = poe[-1][:-1]   
        acceleration[-1] = acceleration[-1][:-1]

        # Create numpy arrays of the data
        acceleration = np.asarray(acceleration,dtype=float)
        poe = np.asarray(poe,dtype=float)

        # Convert probability of exceedence to annual excedence probabilty
        aep = np.divide(np.multiply(-1,np.log(np.subtract(1,poe))),investigation_time)
        
        accel_master_list.append(acceleration)
        aep_master_list.append(aep)
        poe_master_list.append(poe)

    return accel_master_list, aep_master_list, poe_master_list