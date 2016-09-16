#!/usr/bin/env python
import json
import sys
import re
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
from os.path import expanduser

# The extractor(folder_name) function is in 'home/olabiyi'. Please set sys.path before using this function or script if passing pickle file argument.
 
sys.path.append(expanduser("~"))
from data_extractor import extractor

def sync(file_name):
	file_name = os.path.abspath(file_name)
	if os.path.isfile(file_name):
		print "Processing", os.path.split(file_name)[1]
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
		os.chdir(os.path.dirname(file_name))
	elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr.pkl'):
		print "Found an existing exctrated pickle file"
		print "Processing", os.path.split(file_name)[1]
		with open(file_name+'/'+os.path.split(file_name)[1]+'_extr.pkl', 'rb') as f:
			data = pickle.load(f)
			os.chdir(os.path.abspath(file_name))
	else:
		data = extractor(file_name)
    # some data are repeated in time.
	print "Starting data synchronization ..."
	frames = [items['data'].groupby(level=0).first() for items in data]
	index = frames[-1].index
	frames_sync = [frame.reindex(index=index, method='ffill') for frame in frames]
	data = pd.concat(frames_sync, axis=1)
	#data.dropna(axis=1,how='all',inplace=True)
	data = pd.concat([data[data.columns[(data >  0).any()]], data[data.columns[(data <  0).all()]]], axis =1)
	return  data


if __name__ == "__main__":
	args = sys.argv[1:]
	file_name = args[0]
	data_out = sync(file_name)
	print "Saving extracted synced data to csv file"
	pickle.dump(data_out, open(os.path.split(os.getcwd())[1]+'_extr_synced.pkl','wb'))
	data_out.index =  data_out.index.astype(np.int64)//10**3
	data_out.index.name = 'timestamp'
	data_out.to_csv(os.path.split(os.getcwd())[1]+'_extr_synced.csv')
	if len(args) > 1:
		print "Preparing the visualization plots"
		data_out.plot(kind='line')
		plt.show()
