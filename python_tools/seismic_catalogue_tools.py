# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:23:30 2021

@author: ErTodd
"""


def get_gcmt_catalogue(gcmt_outfile, catalogue_raw_files, raw_data_dir, search_params):

    import os
    import shutil
    
    minLat = search_params['lat_min']
    maxLat = search_params['lat_max']

    minLon = search_params['lon_min']
    maxLon = search_params['lon_max']

    minYear = 1976
    minMonth = 1
    minDay = 1

    #Change if you are not searching through the present day
    import datetime
    maxYear = datetime.datetime.today().year
    maxMonth = datetime.datetime.today().month
    maxDay = datetime.datetime.today().day # May have to +/- 1 if location is across the dateline

    minMw = search_params['min_magnitude']
    maxMw = 10

    minMs = search_params['min_magnitude']
    maxMs = 10

    minMb = search_params['min_magnitude']
    maxMb = 10

    minDepth = 0
    maxDepth = 1000

    outfile = gcmt_outfile


    if os.path.isfile(os.path.join(raw_data_dir, outfile)) == True:
        print('Outfile exists.')
        run = 1
        if 'GCMT' in catalogue_raw_files:
            shutil.copy(os.path.join(raw_data_dir, outfile),catalogue_raw_files.get('GCMT')[0])
    else:
        #----------------------------------------------------#
        #code starts

        import urllib.request
        # for python 3.X
        import os

        masterList = []

        search =1
        run = 1

        while True:

            link = "https://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT5/form?itype=ymd&yr=" + str(minYear) + "&mo=" + str(minMonth) + "&day=" + str(minDay) + "&otype=ymd&oyr=" + str(maxYear) + "&omo=" + str(maxMonth) + "&oday=" + str(maxDay) + "&jyr=" + str(minYear) + "" \
            "&jday=" + str(minDay) + "&ojyr=" + str(minYear) + "&ojday=" + str(minDay) + "&nday=" + str(minDay) + "&lmw=" + str(minMw) + "&umw=" + str(maxMw) + "&lms=" + str(minMs) + "&ums=" + str(maxMs) + "&lmb=" + str(minMb) + "&umb=" + str(maxMb) + "&llat=" + str(minLat) + "&ulat=" + str(maxLat) + "&llon=" + str(minLon) + "&ulon=" + str(maxLon) + "&lhd=" + str(minDepth) + "&uhd=" + str(maxDepth) + "" \
            "&lts=-9999&uts=9999&lpe1=0&upe1=90&lpe2=0&upe2=90&list=0&" \
            "start=" + str(search)

            #print(link)

            f = urllib.request.urlopen(link) # for python 3.X add urllib.request
            myfile = f.read()

            myfile = str(myfile,'utf-8') # for python 3.X

            #print(myfile)

            newfile = myfile.split('\n')

            del newfile[0:3]
            del newfile[1:8]
            del newfile[2]
            del newfile[8]
            del newfile[9]

            eq_index = []

            for n in range(0,len(newfile)):    
                if newfile[n][0:7] == '<hr><b>':
                    eq_index.append(n)

            for n in range(0,len(eq_index)):
                newfile[eq_index[n]] = newfile[eq_index[n]].replace('<hr><b>','')
                newfile[eq_index[n]] = newfile[eq_index[n]].replace('        </b>','')
                newfile[eq_index[n]]= newfile[eq_index[n]].replace('<p>','')

                newfile[eq_index[n]+1] = ''

                newfile[eq_index[n]+2] = newfile[eq_index[n]+2].replace('<pre>','')


            if run == 1:       
                masterList.extend(newfile[0:6])
                masterList.append(newfile[8])

            masterList.extend(newfile[9:len(newfile)-10])

            try:
                startSearch = newfile[len(newfile)-10].index('start=')+6
                endSearch = newfile[len(newfile)-10].index('>More solutions')-1
                search = int(newfile[len(newfile)-10][startSearch:endSearch])
            except:
                break

            run = run + 1

        #with open('test.csv', 'wb') as csvfile:
        #    spamwriter = csv.writer(csvfile, delimiter=False)
        #    for n in masterList:
        #        spamwriter.writerow(n)

        with open(os.path.join(raw_data_dir, outfile), 'w+') as export:
            for line in masterList:
                print(line)
                #line = bytes(line + '\r\n','utf-8') # for python 3.X
                export.write(line + '\n')
                #export.write(line)    
            export.write("End of events found with given criteria.")

        if 'GCMT' in catalogue_raw_files:
            shutil.copy(os.path.join(raw_data_dir, outfile),catalogue_raw_files.get('GCMT')[0])

    return run

def magnitude_conversions(scale, mag):
    
    from math import exp
     # mb convert  to mw using exponential models from ISC-GEM Report 201201-V01 (Storchak et al. 2012)
    if scale == "mb" or scale == "mb1mx" or scale == "mbtmp" or scale == "mb1":
        if mag > 6.8:
            mag = mag #NO - this completely ignores mb saturation and will underestimate Mw proxy values
            scale = "mw"
        elif mag > 5 and mag <= 6.8:
            mag = round(exp(-4.66+0.86*mag) + 4.56,1)
            scale = "mw"
        else:
            mag = round(0.85*mag+1.03,1)
            scale = "mw"
           
    # ms convert to mw using exponential models from ISC-GEM Report 201201-V01 (Storchak et al. 2012)
    #mv is a surface wave magnitude defined by Vanek et al. ( 1962)
    elif scale == "ms" or scale == "ms1mx" or scale == "ms7" or scale == "ms1" or scale == "msz" or scale == "mv":     
        mag = round(exp(-0.22+0.23*mag) + 2.86,1) #checked against Storchak et al. 2012 on 9/3/2020
        scale = "mw"           
   
    #convert mags to mw using Rong et al. 2015 eqns
    elif scale == "ml" or scale == "mlv":
        mag = round(1.3210*mag - 1.8631, 1)
        scale = "mw"
        
    # convert Brazillian mR to mw following de Almeida et al. 2018
    elif scale == 'mr':
        mag = mag + 0.34
        scale = 'mw'
       
    #take other magnitudes as Mw
    elif scale == "m" or scale == "md" or scale == "uk" or scale == "mg" or scale == "mu" or scale == "unk" or scale == 'mind' or scale == 'ma' or scale == 'mm' or scale == 'mi':
        mag = mag
        scale = "mw"
       
    #Japan magnitude scale from https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2017JB014697
    elif scale == "mj":
            a = 0.053
            b = 0.33
            c = 1.68
            mag = round(a*mag**2 + b*mag + c ,1)
       
    elif scale == "mw" or scale == "mw(mb)":
            pass
        
    return scale, mag

def organise_catalogues(Compile=1,Duplicate=1):
    
    #Create lists for TimeVal function
    #Number of seconds from beginning of year to beginning of month (non-leap year)
    monthlengths=[31,28,31,30,31,30,31,31,30,31,30]
    for i in range(1,12):
        msec.append(msec[i-1]+monthlengths[i-1]*24*60**2)
    #Number of seconds from beginning of year1 to beginning of years after (beginning with year1)
    for i in range(year1,yearlast):
        ysec.append(ysec[i-year1]+365*24*60**2)
        if ((i%4==0 and i%100!=0) or i%400==0):
            ysec[i-year1+1]=ysec[i-year1+1]+24*60**2
    #

    # Compile: extract eq data from verious agencies
    # Duplicate: locate duplicate events among various events
    time_min=TimeVal(year1,1,1,0,0,0)
    time_max=TimeVal(yearlast,1,1,0,0,0)
#    lat_min,lat_max = -17, 3 #-43, -23  #-7, 5
#    lon_min,lon_max = -82.5, -68.5 #-81, -61 #115, 128
    m_min = min_magnitude
    m_max = 10
    #
    if Compile==1:

        infile=catalogue_raw_files.get('BRAZIL')[0]
        outfile=catalogue_raw_files.get('BRAZIL')[1]
        OrganizeBRAZIL(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)

        #infile=catalogue_raw_files.get('Chile')[0]
        #outfile=catalogue_raw_files.get('Chile')[1]
        #OrganizeCHILE(infile,outfile,chile_lat_min,chile_lat_max,lon_min,lon_max,m_min)
        
        #VELOSO format is copied from SARA
        infile=catalogue_raw_files.get('VELOSO')[0]
        outfile=catalogue_raw_files.get('VELOSO')[1]
        OrganizeSARA(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
        
        infile=catalogue_raw_files.get('SARA')[0]
        outfile=catalogue_raw_files.get('SARA')[1]
        OrganizeSARA(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)

        infile=catalogue_raw_files.get('ANSS')[0]
        outfile=catalogue_raw_files.get('ANSS')[1]
        OrganizeANSS(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)


        infile=catalogue_raw_files.get('GCMT')[0]
        outfile=catalogue_raw_files.get('GCMT')[1]
        OrganizeCMT(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)

        infile=catalogue_raw_files.get('ISCGEM')[0]
        outfile=catalogue_raw_files.get('ISCGEM')[1]
        OrganizeISC(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)

        infile=catalogue_raw_files.get('GEM_Historic')[0]
        outfile=catalogue_raw_files.get('GEM_Historic')[1]
        OrganizeGEMHistoric(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)

        infile=catalogue_raw_files.get('Centennial')[0]
        outfile=catalogue_raw_files.get('Centennial')[1]
        OrganizeCentnnl(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
        
#        infile="RawData/NOAA_catalog_use.csv"
#        outfile="OrganizedData/8-NOAA_OUT.txt"
#        OrganizeNOAA(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
        
#        infile="RawData/Iris_cat_OTMine_08012018_use.csv"
#        outfile="OrganizedData/2-IRIS_OUT.txt"
#        OrganizeIRIS(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
        
#                
#        infile="RawData/GEM_global_historical_eqks.csv"
#        outfile="OrganizedData/3-GEMHistoric_OUT.txt"
#        OrganizeGEMHistoric(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#        
##        infile="RawData/SCR_Catalog_Original.csv"
##        outfile="OrganizedData/6-SCR_OUT.txt"
##        OrganizeOthers(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#
#        
##        infile="RawData/ISC_catalog_OTmine_07312018.csv"
##        outfile="OrganizedData/8-ISC_OUT.txt"
##        OrganizeISC2(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#
#        infile="RawData/NOAA_catalog_use.csv"
#        outfile="OrganizedData/5-NOAA_OUT.txt"
#        OrganizeNOAA(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#       
#        infile="RawData/GFZmt_cat_OTmine_07312018.csv"
#        outfile="OrganizedData/11-GeoFon_OUT.txt"
#        OrganizeOthers(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#        
#        infile="RawData/ChinaUniformedEarthquakeData_use.csv"
#        outfile="OrganizedData/12-China1_OUT.txt"
#        OrganizeOthers2(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#        
#        infile="RawData/ChinaEarthquakeData_use.csv"
#        outfile="OrganizedData/13-China2_OUT.txt"
#        OrganizeOthers2(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#        
#        infile="RawData/XuEtal_2016_use.csv"
#        outfile="OrganizedData/14-XuEtal2016_OUT.txt"
#        OrganizeOthers(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)
#        
#        infile="RawData/EMSC_cat_OTmine_20180802.csv"
#        outfile="OrganizedData/15-EMSC_OUT.txt"
#        OrganizeOthers2(infile,outfile,lat_min,lat_max,lon_min,lon_max,m_min)



 
    if Duplicate==1:
        
        t_window=15 #s
        d_window=100 #km
        m_window=1 #magnitude
        infile_list = [x[1] for x in catalogue_raw_files.values()]
#        infile_list = ["OrganizedData/1-Peru_OUT.txt",
#                       "OrganizedData/2-Chile_OUT.txt",
#                       "OrganizedData/3-SARA_OUT.txt",
#                       "OrganizedData/4-ANSS_OUT.txt",
#                       "OrganizedData/5-GCMT_OUT.txt",
#                       "OrganizedData/6-ISCGEM_OUT.txt",
#                       "OrganizedData/7-GEMHistoric_OUT.txt",
#                       "OrganizedData/8-Centennial_OUT.txt",                     
#                       "OrganizedData/8-NOAA_OUT.txt"
#                       "OrganizedData/11-GeoFon_OUT.txt",
#                       "OrganizedData/12-China1_OUT.txt",
#                       "OrganizedData/13-China2_OUT.txt",
#                       "OrganizedData/14-XuEtal2016_OUT.txt",
#                       "OrganizedData/15-EMSC_OUT.txt"
#                       ]
        outfile = combined_catalogue_file
        CombineWithoutDuplicates(infile_list,outfile,t_window,d_window,m_window)
#        outfile = "Tenke-Combined-nodups-oldmethod.csv"
#        DuEvent(infile_list,outfile,t_window,d_window,m_window)

def plot_MFD_histogram(mag_data, minm, maxm, mind, maxd, maxdt, savedir, plt_save, catname,ftsize):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    binwidth=0.1
    good_bins = np.arange(minm, maxm + binwidth, binwidth)
    # plot cumulative histogram for nsha18 catalogue
    plt.figure(figsize=[5,3.5])
    font = {'family' : 'DejaVu Sans', 'size'   : ftsize}
    plt.rc('font', **font)
    n, bins, patches = plt.hist(mag_data, bins=good_bins, cumulative=-1, color='blue')
    if catname == 'nsha18':    
        if maxdt == 2017:
            print('Plotting MFD for',mind,'to',maxd,'km distance band')
            plt.title('NSHA18 MFD')
            figname = 'NSHA18_MFD_'+str(mind)+'_'+str(maxd)+'_histogram.png'
        else:
            print('Plotting MFD for',mind,'to',maxd,'km distance band')
            plt.title('Post-NSHA18 MFD')
            figname = 'PostNSHA18_MFD_'+str(mind)+'_'+str(maxd)+'_histogram.png'

    elif catname == 'SHEEF2010':
        if maxdt == 2010:
            print('Plotting MFD for',mind,'to',maxd,'km distance band')
            plt.title('SHEEF 2010 MFD')
            figname = 'SHEEF2010_MFD_'+str(mind)+'_'+str(maxd)+'_histogram.png'
        else:
            print('Plotting MFD for',mind,'to',maxd,'km distance band')
            plt.title('Post-SHEEF 2010 MFD')
            figname = 'PostSHEEF2010_MFD_'+str(mind)+'_'+str(maxd)+'_histogram.png'
    elif catname == 'usgs18':    
        if maxdt == 2018:
            print('Plotting MFD for',mind,'to',maxd,'km distance band')
            plt.title('USGS18 MFD')
            figname = 'USGS18_MFD_'+str(mind)+'_'+str(maxd)+'_histogram.png'
        else:
            print('Plotting MFD for',mind,'to',maxd,'km distance band')
            plt.title('Post-USGS18 MFD')
            figname = 'PostUSGS18_MFD_'+str(mind)+'_'+str(maxd)+'_histogram.png'
    else:
        print('Plotting',catname,'catalogue')
        plt.title(catname + ' MFD')
        figname = (catname + '_MFD_'+str(mind)+'_'+str(maxd)+'_histogram.png')


    plt.xlabel('Magnitude', weight='heavy', fontsize=ftsize)
    plt.ylabel('Cumulative number of events', weight='heavy', fontsize=ftsize)
    if plt_save:
        plt.savefig(os.path.join(savedir,figname), format='png', bbox_inches='tight', dpi=300)
    return n, bins

def plot_MFDs(good_bins, nyr, npostyr, sum_nyr_npostyr, figpath, plt_save, catname, ftsize):
    import matplotlib.pyplot as plt
       
    # plot the annualised MFD for each catalogue and their sum
    plt.figure(figsize=[10,7])
    font = {'family' : 'DejaVu Sans', 'size'   : ftsize}
    plt.rc('font', **font)
    ax = plt.gca()
    ax.set_xlim([good_bins[0],good_bins[-1]])
    #ax.set_title('nsha18 Magnitude Frequency Distribution for ' + str(mind/1000) + ' km to ' + str(maxd/1000) + ' km range')
    ax.set_xlabel('Magnitude (Mw)', weight='heavy', fontsize=ftsize)
    ax.set_ylabel('Cumulative occurrence frequency (N/yr)', weight='heavy', fontsize=ftsize)
    #xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    #yticklabels = [item.get_text() for item in ax.get_yticklabels()]
    #ax.set_xticklabels(xticklabels, fontsize=ftsize)
    #ax.set_yticklabels(yticklabels, fontsize=ftsize)
    ax.plot(good_bins[:-1], nyr,c='orange', label=catname.upper()+' catalogue')
    ax.plot(good_bins[:-1], npostyr, c='purple', marker='.', linestyle='--', label='post-'+catname.upper()+' catalogue')
    ax.plot(good_bins[:-1], sum_nyr_npostyr, c='black', linestyle='-.', label='summed catalogues')
    ax.legend(fontsize=ftsize)
    ax.set_yscale('log')
    if plt_save:
        plt.savefig(figpath, format='png', bbox_inches='tight', dpi=300)