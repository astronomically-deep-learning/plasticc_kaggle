#!/usr/bin/python3
"""
 Description / 
 A fast script to read data from the csv file and 
 creat separate files (numpy arrays) for each source per passband.
 NOTE: The class for the training sample is added from 
       the meta info later, so as not to be confused 
       with the test data

 Use /
 ./read_objects s/m
        for single csv file or multiple (individual) files

 History / 
 2018-11-12 v1: reading data from the training_set.csv 
 2018-12-17 v2: generalizing the script to add multiple (individual csv files), reading/writing from/to different directories

"""

###################
# IMPORTING STUFF #
###################
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

opt = sys.argv[1]

#######################
# EDIT PATHS FOR DATA #
#######################

# for one csv file

#path_to_data = "./"             


# for multiple files

#path_to_data = "tmp/"           # testing dir
path_to_data = 'test_data/'     # real dir 


# path to results
path_to_save = "test_data_npy/"

if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)


###############################
# IF ALL DATA IN ONE csv file #
###############################

if opt=='s' or opt=='S':
        df = pd.read_csv('sample_of_training_set.csv')
#        df = pd.read_csv('data/training_set.csv')

        ids = list(set(df['object_id']))
        bands = list(set(df['passband']))

        total = len(ids)
        #print ids
        #print bands

        print(" > Working on: ")
        k = 1	# counter for output printing
        for i in sorted(ids):
	        print(" -- id: {} [ {} / {} ]".format(i, k, total))
	        for b in bands:
        #		print "   ... passband: {}".format(b)
		        obj = df.ix[(df['object_id']==i) & (df['passband']==b)]
		        
		        mjd = obj['mjd']
		        flx = obj['flux']
		        flx_err = obj['flux_err']
		        det = obj['detected']

		        flname = "id-{}_band-{}".format(i,b)
        #		np.savetxt(path_to_save+flname+'.txt', zip(mjd, flx, flx_err, det), fmt=['%5.4f','%.6f','%.6f','%1d'])
		        np.save(path_to_save+flname, zip(mjd, flx, flx_err, det))

	        k += 1


#################################
# ELSE IF MULTIPLE SINGLE FILES #
#################################
# eg from Jeff's script that splits 
# the test_set.csv file

if opt=='m' or opt=='M':
        bands = np.arange(6)
        ids = sorted(glob.glob(path_to_data+'*.csv'))
        total = len(ids)
        print(" > Working on: ")
        k = 1	# counter for output printing
        for idi in ids:
                i = idi.split('_')[-1].strip('.csv')
                print(" -- id: {} [ {} / {} ]".format(i, k, total))
                df = pd.read_csv(idi,
                        names=['object_id','mjd','passband','flux','flux_err','detected'],
                        skiprows=0)

                for b in bands:
                        #print("   ... passband: {}".format(b))
                        obj = df.ix[df['passband']==b]
		        
                        mjd = obj['mjd']
                        flx = obj['flux']
                        flx_err = obj['flux_err']
                        det = obj['detected']

                        flname = "id-{}_band-{}".format(i,b)
                        #np.savetxt(path_to_save+flname+'.txt', zip(mjd, flx, flx_err, det), fmt=['%5.4f','%.6f','%.6f','%1d'])
                        np.save(path_to_save+flname, zip(mjd, flx, flx_err, det))

                k += 1

else:
        sys.exit("!ERROR: not a correct option ('s' or 'm'). Try again.")

#plt.plot(mjd, flx, 'bo')
#plt.show()
