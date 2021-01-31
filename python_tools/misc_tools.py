# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:56:48 2021

@author: ErTodd
"""

def yrp2poe(yrp_list, investigation_time):
    '''
    Convert from year return period to probability of exceedence.

    Parameters
    ----------
    yrp_list : LIST
        List of year (int or float) return periods from which to calculate probability of exceedance.
    investigation_time : INT
        Number of years for which to calculate probability of exceedance.

    Returns
    -------
    poe_list : LIST
        List of corresponding probability of exceedances.

    '''
    import numpy as np
    
    poe_list = list(np.around(-np.expm1(-investigation_time / np.asarray(yrp_list)),decimals=5))
    
    return poe_list

def poe2yrp(poe_list, investigation_time):
    '''
    Convert from year return period to probability of exceedence.

    Parameters
    ----------
    poe_list : LIST
        List of probability of exceedance values (float) from which to calculate return periods.
    investigation_time : INT
        Number of years for which to calculate return period.

    Returns
    -------
    yrp_list : LIST
        List of corresponding return periods.

    '''
    import numpy as np
    
    yrp_list = [int(x) if x < 9000 else int(np.around(x,decimals=-1)) 
                for x in list(np.around(-investigation_time/np.log(-1*np.asarray(poe_list)+1),decimals=0))]
    
    return yrp_list

def openquake_header_checks(header,check1,check2):
    '''
    Parse results header and check that correct results are read

    Parameters
    ----------
    header : STR
        Header line from results file.
    check1 : STR
        Desired IMT.
    check2 : STR
        Mean hazard ('mean').
    '''
    import sys
    
    header = [x.replace('"','').strip() for x in header.split(',')]
    if check1 != 'uhrs':
        imt_ck = [x.split('=')[1].strip("'") for x in header if x.startswith('imt')][0]
        if not imt_ck == check1:
            print('Results read not for',check1)
            sys.exit()
    
    kind_ck = [x.split('=')[1].strip("'") for x in header if x.startswith('kind')][0]
    if not kind_ck == check2:
        print('Results read not for', check2)
        sys.exit()
        
def command_line_git_commit_test():
    import datetime
    print("Testing git commits from command line.",datetime.now())

def my_test_function():
    print('Test this function')
