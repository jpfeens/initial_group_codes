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
        
def read_openquake_source_model_input(source_model_input_filepath):
    '''
    Parse OpenQuake seismic source model input file and put information in a dataframe

    Parameters
    ----------
    source_model_input_filepath : STR
        Filepath of the OpenQuake seismic source model logic tree input file.

    Returns
    -------
    df_source_model : Pandas DataFrame
        DataFrame with the source model branch ID, source model file name, and source model weight.

    '''
    import pandas as pd
    
    df_source_model = pd.DataFrame()
    f = open(source_model_input_filepath)
    data = f.readlines()
    f.close()
    
    branch_id = []
    source_model_name = []
    source_model_weight = []
    
    for line in data:
        if 'branchID=' in line:
            #print(n)
            b = line.split('=')[1]
            branch_id.append(b.split('"')[1])
    
        if  'uncertaintyModel' in line:
            #print(n)
            smn = line.split('</')[0].split('>')[1]
            source_model_name.append(smn)
    
        if  'uncertaintyWeight' in line:
            #print(n)
            smw = line.split('</')[0].split('>')[1]
            source_model_weight.append(smw)

    df_source_model['branch_id'] = branch_id
    df_source_model['source_model'] = source_model_name
    df_source_model['source_model_weight'] = source_model_weight
    
    return df_source_model

def read_openquake_gmm_input(gmm_input_filepath):
    '''
    Parse OpenQuake ground motion model input file and put information in a dataframe

    Parameters
    ----------
    gmm_input_filepath : STR
        Filepath of the OpenQuake ground motion model logic tree input file.

    Returns
    -------
    df_gmms : Pandas DataFrame
        DataFrame with the gmm branch ID, gmm file name, and gmm weight.

    '''
    import pandas as pd
    
    df_source_model = pd.DataFrame()
    f = open(source_model_input_filepath)
    data = f.readlines()
    f.close()
    
    branch_id = []
    source_model_name = []
    source_model_weight = []
    
    for line in data:
        if 'branchID=' in line:
            #print(n)
            b = line.split('=')[1]
            branch_id.append(b.split('"')[1])
    
        if  'uncertaintyModel' in line:
            #print(n)
            smn = line.split('</')[0].split('>')[1]
            source_model_name.append(smn)
    
        if  'uncertaintyWeight' in line:
            #print(n)
            smw = line.split('</')[0].split('>')[1]
            source_model_weight.append(smw)

    df_source_model['branch_id'] = branch_id
    df_source_model['source_model'] = source_model_name
    df_source_model['source_model_weight'] = source_model_weight
    
    return df_source_model