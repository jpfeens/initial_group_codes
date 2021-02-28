# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:28:33 2021

@author: ErTodd
"""

def calculate_ground_motions(gmm_list, desired_calculations, selected_fault, sctx, rctx, dctx, imts, stddev_types):
    '''
    Calculates the expected ground motion and uncertainty, organised by GMPE
    and intensity measure type (i.e. PGA, SA etc.), for a given rupture-site configuration

    Parameters
    ----------
    gmm_list : LIST
        List of openquake GMM classes. Obtained from get_available_gsims().
    desired_calculations : LIST
        List of the calculation outputs deisred. e.g. ['mean', 'stddevs', 'mean_plus_1sd', 'mean_minus_1sd']
    selected_fault : STR
        Fault name for calculation.
    sctx : Class
        OpenQuake SitesContext class containing site information. e.g., Vs30, Z1pt0, etc.
    rctx : Class
        OpenQuake RuptureContext class containing rupture information. e.g. Mmax, rake, depth, dip, ztor, etc.
    dctx : Class
        Openquake DistancesContext class containing distance information. e.g., Rjb, Rrup, Rhypo, Rx, etc.
    imts : LIST
        List of OpenQuake intensity measure classes. e.g., PGA(), SA()
    stddev_types : LIST
        List containing the OpenQuake const.StdDev standard deviation types to calculate. 
        e.g., [const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]

    Returns
    -------
    gmm_ground_motions : DICT
        Dictionary containing calculated ground motion values for the desired calculations 
        and each intensity measure type in imts.

    '''

    import numpy as np
    gmm_ground_motions = dict.fromkeys([str(x) for x in gmm_list])
    nper = len(imts)
    for gmm in gmm_list:
        gmm_ground_motions[gmm] = {"mean": np.zeros(nper),
                                  "stddevs": np.zeros(nper),
                                  "mean_plus_1sd": np.zeros(nper),
                                  "mean_minus_1sd": np.zeros(nper)}            
            
        for i, imt in enumerate(imts):
            try:
                mean, [stddev] = gmm.get_mean_and_stddevs(sctx, rctx, dctx, imt, stddev_types)
                gmm_ground_motions[gmm]['mean'][i] = np.exp(mean)
                gmm_ground_motions[gmm]['stddevs'][i] = stddev
                gmm_ground_motions[gmm]['mean_plus_1sd'][i] = np.exp(mean + stddev)
                gmm_ground_motions[gmm]['mean_minus_1sd'][i] = np.exp(mean - stddev)

            except KeyError:
                # If it raises the error we have just seen (a KeyError) - put in nans
                gmm_ground_motions[gmm]['mean'][i] = np.nan
                gmm_ground_motions[gmm]['stddevs'][i] = np.nan
                gmm_ground_motions[gmm]['mean_plus_1sd'][i] = np.nan
                gmm_ground_motions[gmm]['mean_minus_1sd'][i] = np.nan
                
    return gmm_ground_motions