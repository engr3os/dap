# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 18:21:00 2016

@author: vijay
"""

import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

source_path = '../brain4cars_data/'
save_path = '../brain4cars_data/npy/'

if not os.path.isdir(save_path[:-1]):
	os.makedirs(save_path[:-1])

files = os.listdir(source_path)
dataFiles = []
labels = {'lturn':0, 'rturn':1, 'lchange':2, 'rchange':3, 'straight':4}
trainToTestRatio = 0.7

for file in files:
    if 'csv' in file and 'lock' not in file:
        dataFiles.append(file)
inputData = []
inputTarget = []
for file in dataFiles:
    data = np.array(pd.read_csv(source_path+file,header=None).values).astype(float)
    data = np.transpose(data,[1,0])
    data = np.reshape(data,[-1,7,13])
    labelname = file[:file.index('_')]
    labeldata = labels[labelname]*np.ones([data.shape[0],data.shape[1]]).astype(int)
    inputData.extend(data)
    inputTarget.extend(labeldata)    

inputData = np.array(inputData)
inputTarget = np.array(inputTarget)
print "Size of inputData: ", inputData.shape
print "Size of inputTarget: ", inputTarget.shape

X_train, X_test, Y_train, Y_test= train_test_split(inputData, inputTarget, train_size=trainToTestRatio, random_state=35)


np.save(save_path+'X.npy',inputData)
np.save(save_path+'Y.npy',inputTarget)
np.save(save_path+'X_train.npy',X_train)
np.save(save_path+'Y_train.npy',Y_train)
np.save(save_path+'X_test.npy',X_test)
np.save(save_path+'Y_test.npy',Y_test)



