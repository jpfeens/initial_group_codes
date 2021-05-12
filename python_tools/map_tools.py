# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:18:24 2021

@author: ErTodd
"""

def plot_basemap(map_extent,projection, figsize=[15,15]):
    '''Plot a basemap'''
    import matplotlib.pyplot as plt
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=projection)#ccrs.TransverseMercator(central_latitude=-45.,central_longitude=167))
    ax.set_extent(map_extent)#, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.8, edgecolor='black', linewidth=0.5)
    return fig,ax