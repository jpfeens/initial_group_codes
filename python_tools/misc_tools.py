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
    import numpy as np
    
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
    df_source_model['source_model_weight'] = np.asarray([float(x) for x in source_model_weight]).astype('float32')
    
    return df_source_model

def read_openquake_gmm_input(gmm_input_filepath, trts,v='3.10'):
    '''
    Parse OpenQuake ground motion model input file and put information in a dataframe

    Parameters
    ----------
    gmm_input_filepath : STR
        Filepath of the OpenQuake ground motion model logic tree input file.
    trts : DICT
        Dictionary of the column headers and applicable tectonic region types for the gmms used. 
        e.g. {'gmms_trt1' : 'cratonic', 'gmms_trt2' : 'non-cratonic'}

    Returns
    -------
    df_gmms : Pandas DataFrame
        DataFrame with the gmm tectonic region, gmm file name, and gmm weight.

    '''
    import pandas as pd
    import numpy as np
    import sys
    
    # Add to this list as we expand OpenQuake capabilities with subduction zones
    acceptable_trts = ['cratonic','non_cratonic','Stable Shallow Crust']
    for k,v in trts.items():
        if not k.startswith('gmms_trt'):
            print('trts keys must start with `gmms_trt`, followed by a number')
            sys.exit()
        if v not in acceptable_trts:
            print(v,'is not a currently supported trt.\nCurrently supported trts:',acceptable_trts)
            sys.exit()
        
    
    
    df_gmm_logic_tree = pd.DataFrame()
    f = open(gmm_input_filepath)
    data = f.readlines()
    f.close()
    
    trts_indices_start = []
    trts_indices_end_all = []
    all_trts = []
    for linenum in range(len(data)):
        if 'applyToTectonicRegionType=' in data[linenum]:
            trts_indices_start.append(linenum)
            all_trts.append(data[linenum].split('"')[1].lower())
        if '</logicTreeBranchSet>' in data[linenum]:
            trts_indices_end_all.append(linenum)
    
    trts_indices_end_all = np.asarray(trts_indices_end_all)
    trts_indices_end = []
    #Clean up end lines that don't correspond with start lines
    if len(trts_indices_end_all) != len(trts_indices_start):
        for i,s in enumerate(trts_indices_start):
            end_ind = [min(x) for x in list(np.subtract(trts_indices_end_all,s)) if x > 0][0]
            trts_indices_end.append(end_ind)
    else:
        trts_indices_end = trts_indices_end_all

   
    # Only keep the tectonic region types applicable to this site
    keep_trts_inds = [(trts_indices_start[i],trts_indices_end[i]) for i,x in enumerate(all_trts) if x.lower() in trts.values()]
    
    gmm_tectonic_region = []
    gmm_name = []
    gmm_weight = []
    
    for i,trt_ind_range in enumerate(keep_trts_inds):
        count = 0
        start_i = trt_ind_range[0]
        end_i = trt_ind_range[1]
        tr = data[start_i].split('"')[1]
        #print('working on',tr,'GMMs from lines',start_i,end_i)
        for line in data[start_i:end_i]:
            if  'uncertaintyModel' in line:
                gmmn = line.split('</')[0].split('>')[1]
                gmm_name.append(gmmn)
                count = count +1
            if  'uncertaintyWeight' in line:
                gmmw = line.split('</')[0].split('>')[1]
                gmm_weight.append(gmmw)
        gmm_tr_tmp = [tr]*count
        gmm_tectonic_region.append(gmm_tr_tmp)
    
    df_gmm_logic_tree['tectonic_region'] = [item for sublist in gmm_tectonic_region for item in sublist]
    df_gmm_logic_tree['gmm_name'] = gmm_name
    df_gmm_logic_tree['gmm_weight'] = np.asarray([float(x) for x in gmm_weight]).astype('float32')
    
    return df_gmm_logic_tree

def format_openquake_realization_output(realisation_output_filepath):
    '''
    Open, parse, and create a DataFrame from the OpenQuake realisation output

    Parameters
    ----------
    realisation_output_filepath : STR
        Filepath to OpenQuake realization csv output file.

    Returns
    -------
    df_realisation : Pandas DataFrame
        DataFrame containing the rlz values, branch paths, and weights for each logic tree realisation.

    '''
    import sys
    import pandas as pd
    import numpy as np
    df_realisation = pd.read_csv(realisation_output_filepath)
    df_realisation['rlz_id'] = ["%03d" % i for i in df_realisation['rlz_id'].tolist()]
    
    # Determine how many tectonic region types were present in the calculation to categorise gmms 
    # e.g. separate cratonic gmms from non-cratonic gmms from subduction gmms
    tmp_branch_path = df_realisation.loc[0,'branch_path']
    num_trts = len(tmp_branch_path.split('~')[1].split('_'))
    
    if num_trts == 0:
        print('No GMMs listed in realization branch path.')
        sys.exit()
    else:
        gmm_colnames = ['gmms_trt'+ str(x) for x in range(1,num_trts+1)]

    for i,gmm_colname in enumerate(gmm_colnames):
        df_realisation[gmm_colname] = [x.split('~')[1].split('_')[i] for x in df_realisation['branch_path'].tolist()]
    
    df_realisation['branch_path'] = [x.split('~')[0] for x in df_realisation['branch_path'].tolist()]
    df_realisation.rename(columns={'weight': 'rlz_weight', 'branch_path' : 'source_model_branch_path'},inplace=True)
    
    # Check that the realisation weights sum to 1
    rlz_weights = df_realisation['rlz_weight'].to_numpy()
    if np.sum(rlz_weights.astype('float32')) != 1.0:
        print('rlz weights do not sum to 1.')
        print(np.sum(rlz_weights))
        sys.exit()
    
    return df_realisation, num_trts

def rename_openquake_realisation_df_cols(df_realisation, num_trts, trts):
    trts_colnames = dict.fromkeys(trts.keys())
    for trt_num,trt in trts.items():
        trts_colnames[trt_num] = trt+'_gmms'
    
    if num_trts != len(trts):
        print("The number of output tectonic region types",num_trts,"doesn't match the number of input tectonic region types",len(trts),'.')
        sys.exit()
    else:
        df_realisation.rename(columns = trts_colnames, inplace=True)
        
    return df_realisation
    

def parse_openquake_job_ini_file(job_input_filepath):
    '''
    Parse OpenQuake job.ini input file and return the return periods, fractiles, and IMTs used in the calculation.

    Parameters
    ----------
    job_input_filepath : STR
        Filepath of job.ini file used in OpenQuake calculations.

    Returns
    -------
    return_periods : LIST
        List of return periods in decimal poe form.
    fractiles : LIST
        List of fractiles in decimal form.
    IMTs : LIST
        List of IMTs used in `PGA` or `SA(0.2)` OpenQuake form.
    investigation_time : INT
        Integer number of years over which the calculation is performed.

    '''
    f = open(job_input_filepath)
    data = f.readlines()
    f.close()

    IMTs_tmp = []
    for linenum in range(len(data)):
        if data[linenum].startswith('quantile_hazard_curves'):
            fractiles = sorted(list(data[linenum].split('=')[1].strip().replace(',','').split(' ')))
        elif data[linenum].startswith('intensity_measure_types_and_levels') or data[linenum].startswith('\t'):
            IMTs_tmp.append(data[linenum])
        elif data[linenum].startswith('investigation_time'):
            investigation_time = int(float(data[linenum].split('=')[1].strip()))
        elif data[linenum].startswith('poes'):
            return_periods = sorted(list(data[linenum].split('=')[1].strip().replace(',','').split(' ')),reverse=True)
    
    # If no fractiles are defined, set fractiles as 'mean' only
    try:
        fractiles
    except NameError:
        fractiles = ['mean']   

    #Clean up IMTs
    if len(IMTs_tmp) == 1:
        IMTs = list(dict(IMTs_tmp.split('=')[1].strip().strip('"')).keys())
    else:
        IMTs=[]
        for imt_line in IMTs_tmp: 
            if imt_line.startswith('intensity_measure_types_and_levels'):
                IMTs.append(imt_line.split('{')[1].split('"')[1])
            else:
                IMTs.append(imt_line.split('"')[1])

    return return_periods, fractiles, IMTs, investigation_time


def make_gmm_label(string_to_parse):
    '''
    Adds spaces before capital letters or numbers. If two words are adjacent, it adds the word 'and' between them

    Parameters
    ----------
    string_to_parse : STR
        String version of GMM name that needs to be converted to a label for a legend entry.

    Returns
    -------
    STR
        GMM label ready for legend entry.

    '''
    import re
    words_or_numbers = re.findall('([A-Z][a-z]*)(\d+)*', string_to_parse)
    words_or_numbers = [item if item != '' else 'and' for sublist in words_or_numbers for item in sublist]
    return ' '.join(words_or_numbers)