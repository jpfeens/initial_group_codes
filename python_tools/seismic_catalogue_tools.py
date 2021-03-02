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

        shutil.copy(os.path.join(raw_data_dir, outfile),catalogue_raw_files.get('GCMT')[0])

    return run