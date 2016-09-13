# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:25:32 2016

@author: vijay
"""

import os
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing


path = '../tripCsvFiles/'
savepath = '../tripCsvFiles/pca/'
sessionIndexFile = "sessionIndex.csv"

sessionIndex = pd.read_csv(sessionIndexFile)

if not os.path.isdir(savepath[:-1]):
    os.makedirs(savepath[:-1])

files = os.listdir(path)
dataFiles = []
badTripsForPCA = ['1459973682141964','1459980768823544']  #the meaning is that these two trips have difference baselines in PCA which contorts the whole PCA data while scaling

for file in files:
    if '_features_brake_agg.csv' in file and 'lock' not in file and file[:file.index('_')] not in badTripsForPCA:
        dataFiles.append(file)

print "Total no. of files found %d" %len(dataFiles)

columns = ["timestamp",
           "speed",
           "engine_rpm",
           "throttle_position",
           "elevation",
           "acceleration",
           "intersectionDist",
           "intersectionDirection",
           "lvelX",
           "lposX",
           "rposX",
           "rvelX",
           "mvelX",
           "mposX",
           "leftlane_valid",
           "leftlane_curvature",
           "rightlane_valid",
           "rightlane_curvature",
           "pbrk",   #take a note about this when copying
           "hv_accp",
           "b_p",
           "numHandsOnWheel","left_x0","left_y0","left_d","left_theta","left_onOff","left_isMoving","left_movingD","left_movingTheta","right_x0","right_y0","right_d","right_theta","right_onOff","right_isMoving","right_movingD","right_movingTheta","dBeweenHands","angleBetweenHands", #hand movement columns
           "binX1","binX2","binX3","binX4","binTheta1","binTheta2","binTheta3","binTheta4","Mean", #face feature columns
           "brakeEvents","sparseBrakeEvents","emptyBrakeEvents"]  #brake events for the target layer

columns_in_data = [x for x in columns if x not in ["timestamp","brakeEvents","sparseBrakeEvents","emptyBrakeEvents","pbrk"]]
columns_not_in_data = ["timestamp","brakeEvents","sparseBrakeEvents","emptyBrakeEvents","pbrk"] #remove to include pbrk in this list

print "Total no. of input features: %d" %(len(columns_in_data)-1) #ofcourse this includes pbrk. hence the -1

inputPCATrainData = []
inputPCATestData = []
inputHolderTrainData = []
inputHolderTestData = []
fileLengthDict = {}
num_components = 20 #number of PCA components
maxPos = 500 #max value of position in object log
counterTrain = 0
counterTest = 0

for file in dataFiles:
    print "Processing file %s" %file
    folder = file[:file.index('_')]
    #train_test_index = sessionIndex.train_or_test_for_pca[sessionIndex.timestamp==int(folder)].values
    """for demo"""
    train_test_index = sessionIndex.train_or_test_for_pca_demo[sessionIndex.timestamp==int(folder)].values


    data = pd.read_csv(path+file, usecols = columns)
    #print data
    #print data[['lposX','mposX','rposX']].max()
    data['lposX'][data['lposX']==0] = maxPos
    data['rposX'][data['rposX']==0] = maxPos
    data['mposX'][data['mposX']==0] = maxPos

    if not train_test_index:   #meaning this is a trip in training set        
        inputPCATrainData.extend(data[columns_in_data].values.tolist())
        inputHolderTrainData.extend(data[columns_not_in_data].values.tolist())
        fileLengthDict[file] = [counterTrain, counterTrain+data.shape[0]]
        counterTrain = counterTrain + data.shape[0]
    else:
        inputPCATestData.extend(data[columns_in_data].values.tolist())
        inputHolderTestData.extend(data[columns_not_in_data].values.tolist())
        fileLengthDict[file] = [counterTest, counterTest+data.shape[0]]
        counterTest = counterTest + data.shape[0]        

print "Started processing PCA and scaling..."
inputPCATrainData = np.array(inputPCATrainData)
inputHolderTrainData = np.array(inputHolderTrainData)
inputPCATestData = np.array(inputPCATestData)
inputHolderTestData = np.array(inputHolderTestData)

"""pca"""
pca = RandomizedPCA(n_components = num_components, whiten = True).fit(inputPCATrainData)
partialTrainDataPCA = pca.transform(inputPCATrainData)
partialTestDataPCA = pca.transform(inputPCATestData)

"""scale between 0 and 1"""
min_max_scaler = preprocessing.MinMaxScaler().fit(partialTrainDataPCA)
partialTrainDataPCA = min_max_scaler.transform(partialTrainDataPCA)
partialTestDataPCA = min_max_scaler.transform(partialTestDataPCA)

new_columns = ['f'+str(i) for i in range(1,num_components+1)]

print "Starting creating pca files...."

for file in dataFiles:
    folder = file[:file.index('_')]
    train_test_index = sessionIndex.train_or_test_for_pca[sessionIndex.timestamp==int(folder)].values
    
    if not train_test_index:   #meaning trip is in training
        data = pd.DataFrame(partialTrainDataPCA[fileLengthDict[file][0]:fileLengthDict[file][1]],columns = new_columns)
        data2 = pd.DataFrame(inputHolderTrainData[fileLengthDict[file][0]:fileLengthDict[file][1]],columns = columns_not_in_data)
    else:
        data = pd.DataFrame(partialTestDataPCA[fileLengthDict[file][0]:fileLengthDict[file][1]],columns = new_columns)
        data2 = pd.DataFrame(inputHolderTestData[fileLengthDict[file][0]:fileLengthDict[file][1]],columns = columns_not_in_data)
    
    data = data.join(data2)
    data = data.set_index('timestamp')
    data.to_csv(savepath + file[:-4] + '_pca.csv')
    print "Processed file %s" %(file[:-4] + '_pca.csv')

"""get the list of all training and testing trip IDs from sessionIndexFile or /home/vijay/scripts/sessionIndex.csv"""

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    


