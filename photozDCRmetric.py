from lsst.sims.maf.metrics import BaseMetric
import astrometry_defs as astr
import numpy as np
from scipy import stats
from astropy.table import Table
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report
from astroML.linear_model import NadarayaWatson

class DCRphotozMetric(BaseMetric):
	def __init__(self, metricName='DCRphotoz', **kwargs): 
		#define column names from opsim that will be available to use in run() 
		self.AMcol = 'airmass' #airmass distrib for sky healpixel
		self.Fcol = 'filter' #band filters (u, g, r, i, z, etc.)
		cols = [self.AMcol, self.Fcol] 
		#call BaseMetric's __init__ to get basic metric functionality down
		super(DCRphotozMetric, self).__init__(col=cols, metricName=metricName, **kwargs) 
	#run() defines what the metric does at each healpix, dataSlice is the opsim table info for each healpixel
	def run(self, dataSlice, slicePoint=None):
		data = Table.read('mastertrainingmatch.fits') #read in quasar data
		#cut out negative fluxes in each filter band
		mask = ( (data['PSFFLUX'][:,0]>0) & (data['PSFFLUX'][:,1]>0) & (data['PSFFLUX'][:,2]>0) & (data['PSFFLUX'][:,3]>0) & (data['PSFFLUX'][:,4]>0)  ) 
		data = data[mask]
		#array for holding dcr slopes
		tempDCRarray = [] 
		#calculate DCR slope for each object in our table
		for x in data['ZSPEC_1']: 
			#calculate tangent of zenith angle and parallactic offset (tan(Z) and R)
			tanZList, RList = astr.calcR(dataSlice[self.AMcol], dataSlice[self.Fcol], zshift = x)
			#calculate a slope and store in tempDCRarray
			slope, intercept, r_value, p_value, std_err = stats.linregress(tanZList, RList)
			tempDCRarray.append(slope)
		#add the column of DCR slopes into our table
		data['DCRSLOPE'] = tempDCRarray
		#this just makes sure all the columns are correctly formatted for vstack
		data = data.filled() 
		#colors data, properly formatted
		X = np.vstack([ data['ug'], data['gr'], data['ri'], data['iz'], data['zs1'], data['s1s2'] ]).T
		#spectroscopic redshift, properly formatted
		y = np.array(data['ZSPEC_1'])
		#split data into 80 percent training, 20 percent testing
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=73)
		#setup NW model w/ gaussian kernel and kernel width 0.05
		model1 = NadarayaWatson('gaussian', 0.05) 
		model1.fit(X_train, y_train) #fit model to training set
		pred1 = model1.predict(X_test) #predict based on fit
		#do a test to see what fraction of points are within 0.1 of being correctly predicted
		#total # of points
		n = len(pred1)
		#is the difference between prediction and actual <0.1?
		mask13 = (np.abs(pred1 - y_test)<0.1)
		#number of points that are within 0.1 of actual value
		m13 = len(pred1[mask13])
		frac13 = 1.0*m13/n #fraction of all points within 0.1 of actual answer
		#colors and DCR, properly formatted
		X2 = np.vstack([ data['ug'], data['gr'], data['ri'], data['iz'], data['zs1'], data['s1s2'], data['DCRSLOPE'] ]).T 
		y2 = np.array(data['ZSPEC_1']) #potentially unnecessary, given existence of y
		#same split as above, so the 4 sets of objects are identical
		X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=73)
		model2 = NadarayaWatson('gaussian', 0.05) #potentially unnecessary, given existence of model1, not sure if model's can be refit safely
		#fit to new training sets
		model2.fit(X2_train, y2_train)
		pred2 = model2.predict(X2_test)
		#same test as above, measure how many predictions are within 0.1
		n = len(pred2)
		mask23 = (np.abs(pred2- y2_test)<0.1)
		m23 = len(pred2[mask23])
		frac23 = 1.0*m23/n
		#fraction of points that moved into within 0.1 w/ DCR training
		improve = frac23 - frac13 
		return improve
