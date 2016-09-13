# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:41:03 2016

@author: vijay
"""

import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing
from sklearn.utils import shuffle
from itertools import chain
import matplotlib.pyplot as plt

def getnorm(train,test,eventWindow,n):
    train = np.reshape(train,(len(train)*eventWindow,n))
    test = np.reshape(test,(len(test)*eventWindow,n))
    min_max_scaler = preprocessing.MinMaxScaler().fit(train)
    train = min_max_scaler.transform(train)
    test = min_max_scaler.transform(test)
    train = np.reshape(train,[-1,eventWindow,n])
    test = np.reshape(test,[-1,eventWindow,n])
    return train, test

path = '../tripCsvFiles/'
savepath_balanced_demo = '../data_for_network_balanced_demo/'

"""set this parameter to include or not include braking zones (zones where the brake pressure is non-zero"""
include_braking_zones = 1

num_components = 20 #the number of components for PCA. This should be equal to or greater than the total number of features
eventWindow = 50 #meaning 5 second window
buffer = 1.5 #meaning we are going to have 1.5 timess the no. of neg samples than positive samples


files = os.listdir(path)
dataFilesTrain = []
dataFilesTest = []
sessionIndexFile = "sessionIndex.csv"

sessionIndex = pd.read_csv(sessionIndexFile)

badTripsForPCA = ['1459973682141964','1459980768823544']  #the meaning is that these two trips have difference baselines in PCA which contorts the whole PCA data while scaling
tripsForTestDemo = sessionIndex.timestamp[sessionIndex.train_or_test_for_pca_demo==1].tolist()
tripsForTestDemo = [str(i).strip() for i in tripsForTestDemo]
print "tripsForTestDemo: ", tripsForTestDemo

for file in files:
    #if '_features_brake_agg_pca.csv' in file and 'lock' not in file and file[:file.index('_')] not in badTripsForPCA:
    if '_features_brake_agg.csv' in file and 'lock' not in file and file[:file.index('_')] not in badTripsForPCA:
        if file[:file.index('_')] not in tripsForTestDemo:
            dataFilesTrain.append(file)
        else:
            dataFilesTest.append(file)
        
print "Total no. of files found for train : %d" %len(dataFilesTrain)
print "Total no. of files found for test : %d" %len(dataFilesTest)

#columns = ["timestamp"]+["f"+str(i) for i in range(1,num_components+1)]+["brakeEvents","sparseBrakeEvents","emptyBrakeEvents","pbrk"]
#columns_in_data = [x for x in columns if x not in ["timestamp","brakeEvents","sparseBrakeEvents","emptyBrakeEvents","pbrk"]] #remember to add pbrk to this list
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
          "pbrk",
          "leftlane_valid",
          "leftlane_curvature",
          "rightlane_valid",
          "rightlane_curvature",
          "hv_accp",
          "b_p",
          "numHandsOnWheel","left_x0","left_y0","left_d","left_theta","left_onOff","left_isMoving","left_movingD","left_movingTheta","right_x0","right_y0","right_d","right_theta","right_onOff","right_isMoving","right_movingD","right_movingTheta","dBeweenHands","angleBetweenHands", #hand movement columns
          "binX1","binX2","binX3","binX4","binTheta1","binTheta2","binTheta3","binTheta4","Mean", #face feature columns
          "brakeEvents","sparseBrakeEvents","emptyBrakeEvents"]  #brake events for the target layer

columns_in_data = [x for x in columns if x not in ["timestamp","brakeEvents","sparseBrakeEvents","emptyBrakeEvents","pbrk"]]

print "Total no. of input features: %d" %len(columns_in_data)


"""creating training data"""

inputMatrixTrue = [] #capture braking events
targetMatrixTrue = [] #capture target for braking events
inputMatrixFalse = [] #capture non braking events
targetMatrixFalse = [] #capture target for non braking events

print "Started creating training data..."
for file in dataFilesTrain:
    print "Processing file %s" %file
    data = pd.read_csv(path+file, usecols = columns)
    inputData = data[columns_in_data].values.tolist()
    brakeEvents = data['sparseBrakeEvents'].values.tolist()
    emptyBrakeEvents = data['emptyBrakeEvents'].values.tolist()

    #slicing 5 sec brake events
    for i in range(len(brakeEvents)):
        if brakeEvents[i] ==1 and i-eventWindow >=0 and max(emptyBrakeEvents[i-eventWindow+1:i+1])<1:
            inputMatrixTrue.append(inputData[i-eventWindow:i])
    #slicing 5 sec non brake events
    for i in range(len(emptyBrakeEvents)):
        if emptyBrakeEvents[i] ==1 and i-eventWindow >=0 and max(brakeEvents[i-eventWindow+1:i+1])<1:
            inputMatrixFalse.append(inputData[i-eventWindow:i])

    print "Done processing file %s" %file
print "Total no. of sparse braking events in Train:%d" %len(inputMatrixTrue)
print "Total no. of non braking events in Train: %d" %len(inputMatrixFalse)

targetMatrixTrue = [[1]*len(inputMatrixTrue[0]) for i in range(len(inputMatrixTrue))]
targetMatrixFalse = [[0]*len(inputMatrixFalse[0]) for i in range(len(inputMatrixFalse))]

#combining, shuffling and splitting data
finalDataMatrix = inputMatrixTrue + inputMatrixFalse
finalTargetMatrix = targetMatrixTrue + targetMatrixFalse
finalDataMatrix, finalTargetMatrix = shuffle(finalDataMatrix,finalTargetMatrix,random_state = 15)
X_train = np.array(finalDataMatrix)
Y_train = np.array(finalTargetMatrix)

print "Shape of X_train:", X_train.shape
train_data = np.reshape(X_train,(len(X_train)*eventWindow,len(columns_in_data)))
pca = RandomizedPCA(n_components = num_components, whiten = True).fit(train_data)
X_train_pca = pca.transform(train_data)
X_train_pca = np.reshape(X_train_pca, (len(X_train),eventWindow,num_components))


ind_train_neg = np.where(Y_train[:,0]==0)[0].astype(int)
pos_train_samples = np.where(Y_train[:,0] == 1)[0].astype(int)
ind_train_neg = shuffle(ind_train_neg)
X_train_balanced = np.delete(X_train,ind_train_neg[len(pos_train_samples)*buffer:],0)
X_train_pca_balanced = np.delete(X_train_pca,ind_train_neg[len(pos_train_samples)*buffer:],0)
Y_train_balanced = np.delete(Y_train,ind_train_neg[len(pos_train_samples)*buffer:],0)
print "Shape of X_train_balanced:",X_train_balanced.shape
print "Shape of Y_train_balanced:",Y_train_balanced.shape





##########################################################################################
##########################################################################################
"""creating test data"""

inputMatrix = []
inputTargetMatrix = []
countBrakeEvents = 0
countNonBrakeEvents = 0
fig = 0
print "Started creating test data..."
testTargetList = []
testDataList = []
for file in dataFilesTest:
    print "Processing file %s" %file
    data = pd.read_csv(path+file, usecols = columns)
    inputData = data[columns_in_data].values.tolist()
    testTargetListPerTrip = [None]*len(inputData)
    brakePressure = data['pbrk'].values.tolist()

    #slicing 5 sec brake events
    brakeEvents = data['sparseBrakeEvents'].values.tolist()
    #slicing 5 sec non brake events
    emptyBrakeEvents = data['emptyBrakeEvents'].tolist()
    for i in range(len(brakeEvents)):
        if brakeEvents[i] ==1 and i-eventWindow >=0 and 0 not in testTargetListPerTrip[i-eventWindow:i] and 1 not in testTargetListPerTrip[i-eventWindow:i]: # the last part is to make sure event and no event zones dont overlap
            inputMatrix.append(inputData[i-eventWindow:i])
            inputTargetMatrix.append([1]*eventWindow)
            testTargetListPerTrip[i-eventWindow:i] = [1]*eventWindow
            countBrakeEvents +=1
            
        elif emptyBrakeEvents[i] ==1 and i-eventWindow >=0 and 1 not in testTargetListPerTrip[i-eventWindow:i] and 0 not in testTargetListPerTrip[i-eventWindow:i]: # the last part is to make sure event and no event zones dont overlap
            inputMatrix.append(inputData[i-eventWindow:i])
            inputTargetMatrix.append([0]*eventWindow)
            testTargetListPerTrip[i-eventWindow:i]=[0]*eventWindow
            countNonBrakeEvents +=1
          
    sparseZeroCros = np.array([i for i, x in enumerate(brakeEvents) if x == 1])
    emptypoints = np.array([i for i, x in enumerate(emptyBrakeEvents) if x == 1])
    brakePressure = np.array(brakePressure)
    """
    plt.figure(fig)        
    plt.plot(brakePressure, 'g-')
    plt.plot(sparseZeroCros,brakePressure[sparseZeroCros], 'r*', markersize=10)
    plt.title('brake pressure sparse zero crossing')
    
    plt.figure(fig)        
    plt.plot(brakePressure, 'g-')
    plt.plot(emptypoints, brakePressure[emptypoints], 'g*', markersize=10)
    plt.title('brake pressure empty points')
    fig+=1
    """
    
      
    testDataListPerTrip = [inputData[i*eventWindow:(i+1)*eventWindow] for i in range(int(len(inputData)/eventWindow))] #creating test data with braking zones

    print np.array(testDataListPerTrip).shape

    testTargetListPerTrip = np.array(testTargetListPerTrip[:len(testDataListPerTrip)*eventWindow])
    testTargetListPerTrip = np.reshape(testTargetListPerTrip,[-1, eventWindow])

    print "Sum of testTargetListPerTrip :", sum([i for i in testTargetListPerTrip.flatten() if i==0 or i==1])
    print "Sum of testTargetListPerTrip only 0 :", sum([1 for i in testTargetListPerTrip.flatten() if i==0])
    
    testDataList.extend(testDataListPerTrip)
    testTargetList.extend(testTargetListPerTrip)
    print "Shape of testDataList: ", np.array(testDataList).shape
    print "Shape of testTargetList: ",np.array(testTargetList).shape 

    print "Done processing file %s" %file

print "Total no. of sparse braking events in Test:%d" %countBrakeEvents
print "Total no. of non braking events in Test: %d" %countNonBrakeEvents



"""here we decide what data to create depending on whether to include or not include braking zones"""
if include_braking_zones:
    X_test = np.array(testDataList)
    Y_test = np.array(testTargetList)
else:
    X_test = np.array(inputMatrix)
    Y_test = np.array(inputTargetMatrix)

print "Shape of X_test:", X_test.shape
print "Shape of Y_test", Y_test.shape

test_data = np.reshape(X_test,(len(X_test)*eventWindow,len(columns_in_data)))
X_test_pca = pca.transform(test_data)
X_test_pca = np.reshape(X_test_pca,(len(X_test),eventWindow,num_components))

"""scaling train and test PCA data"""
#scaling data between 0 and 1
X_train_pca_balanced_norm, X_test_pca_norm = getnorm(X_train_pca_balanced, X_test_pca,eventWindow,num_components)
X_train_balanced_norm, X_test_norm = getnorm(X_train_balanced, X_test, eventWindow,len(columns_in_data))
X_train_balanced, X_train_balanced_norm, X_train_pca_balanced, X_train_pca_balanced_norm, Y_train_balanced = shuffle(X_train_balanced, X_train_balanced_norm, X_train_pca_balanced, X_train_pca_balanced_norm, Y_train_balanced, random_state = 15)

np.save(savepath_balanced_demo+'X_train_20pca_balanced.npy',X_train_pca_balanced)
np.save(savepath_balanced_demo+'X_test_20pca_balanced.npy',X_test_pca)
np.save(savepath_balanced_demo+'X_train_20pca_balanced_norm.npy',X_train_pca_balanced_norm)
np.save(savepath_balanced_demo+'X_test_20pca_balanced_norm.npy',X_test_pca_norm)
np.save(savepath_balanced_demo+'Y_train_balanced.npy',Y_train_balanced)
np.save(savepath_balanced_demo+'Y_test_balanced.npy',Y_test)
np.save(savepath_balanced_demo+'X_train_balanced.npy',X_train_balanced)
np.save(savepath_balanced_demo+'X_test_balanced.npy',X_test)
np.save(savepath_balanced_demo+'X_train_balanced_norm.npy',X_train_balanced_norm)
np.save(savepath_balanced_demo+'X_test_balanced_norm.npy',X_test_norm)



