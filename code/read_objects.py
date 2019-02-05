#!/usr/bin/python
"""
 A fast script to read data from the csv file and
 creat separate files for each source per passband.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data = np.genfromtxt('sample_of_training_set.csv', delimiter=',', names=True )
#print data
#print data['object_id']



# df = pd.read_csv('../data/sample_of_training_set.csv')
df = pd.read_csv('../data/' + sys.argv[1])
#df = pd.read_csv('training_set.csv')
#print type(df)

ids = list(set(df['object_id']))
bands = list(set(df['passband']))

total = len(ids)
#print ids
#print bands

### The class is added from the meta info later
### so as not to be confused with the test data

print(" > Working on: ")
k = 0	# counter for output printing
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
		np.savetxt(flname+'.txt', zip(mjd, flx, flx_err, det), fmt=['%5.4f','%.6f','%.6f','%1d'])
#		np.save(flname, zip(mjd, flx, flx_err, det))

	k += 1


#plt.plot(mjd, flx, 'bo')
#plt.show()
