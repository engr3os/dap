# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:53:01 2016

@author: vijay
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing



path = '../tripCsvFiles/'
savepath = '../data_for_network/'
savepath_balanced = '../data_for_network_balanced/'

if not os.path.isdir(savepath[:-1]):
  os.makedirs(savepath[:-1])
if not os.path.isdir(savepath_balanced[:-1]):
  os.makedirs(savepath_balanced[:-1])

files = os.listdir(path)
dataFiles = []

badTripsForPCA = ['1459973682141964','1459980768823544']  #the meaning is that these two trips have difference baselines in PCA which contorts the whole PCA data while scaling

for file in files:
    if '_features_brake_agg.csv' in file and 'lock' not in file and file[:file.index('_')] not in badTripsForPCA:
        dataFiles.append(file)
        
print "Total no. of files found %d" %len(dataFiles)

def getnorm(train,test,eventWindow,n):
    train = np.reshape(train,(len(train)*eventWindow,n))
    test = np.reshape(test,(len(test)*eventWindow,n))
    min_max_scaler = preprocessing.MinMaxScaler().fit(train)
    train = min_max_scaler.transform(train)
    test = min_max_scaler.transform(test)
    train = np.reshape(train,[-1,eventWindow,n])
    test = np.reshape(test,[-1,eventWindow,n])
    return train, test

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
           "hv_accp",
           "b_p",
           "numHandsOnWheel","left_x0","left_y0","left_d","left_theta","left_onOff","left_isMoving","left_movingD","left_movingTheta","right_x0","right_y0","right_d","right_theta","right_onOff","right_isMoving","right_movingD","right_movingTheta","dBeweenHands","angleBetweenHands", #hand movement columns
           "binX1","binX2","binX3","binX4","binTheta1","binTheta2","binTheta3","binTheta4","Mean", #face feature columns
           "brakeEvents","sparseBrakeEvents","emptyBrakeEvents"]  #brake events for the target layer

columns_in_data = [x for x in columns if x not in ["timestamp","brakeEvents","sparseBrakeEvents","emptyBrakeEvents"]]

print "Total no. of input features: %d" %len(columns_in_data)

inputMatrixTrue = [] #capture braking events
targetMatrixTrue = [] #capture target for braking events
inputMatrixFalse = [] #capture non braking events
targetMatrixFalse = [] #capture target for non braking events

eventWindow = 50 #meaning 5 second window
num_components = 20  #the number of components for PCA. This should be equal to or greater than the total number of features
maxPos=500

for file in dataFiles:
    print "Processing file %s" %file
    data = pd.read_csv(path+file, usecols = columns)
    #"""
    data['lposX'][data['lposX']==0] = maxPos
    data['rposX'][data['rposX']==0] = maxPos
    data['mposX'][data['mposX']==0] = maxPos
    #"""
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
print "Total no. of sparse braking events :%d" %len(inputMatrixTrue)
print "Total no. of non braking events: %d" %len(inputMatrixFalse)

targetMatrixTrue = [[1]*len(inputMatrixTrue[0]) for i in range(len(inputMatrixTrue))]
targetMatrixFalse = [[0]*len(inputMatrixFalse[0]) for i in range(len(inputMatrixFalse))]

#combining, shuffling and splitting data
finalDataMatrix = inputMatrixTrue + inputMatrixFalse
finalTargetMatrix = targetMatrixTrue + targetMatrixFalse
trainToTestRatio = 0.8 #test ratio = 1-trainToTestRatio

X_train, X_test, Y_train, Y_test= train_test_split(finalDataMatrix, finalTargetMatrix, train_size=trainToTestRatio, random_state=15)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

train_data = np.reshape(X_train,(len(X_train)*eventWindow,len(columns_in_data)))
test_data = np.reshape(X_test,(len(X_test)*eventWindow,len(columns_in_data)))
pca = RandomizedPCA(n_components = num_components, whiten = True).fit(train_data)
X_train_pca = pca.transform(train_data)
X_test_pca = pca.transform(test_data)
X_train_pca = np.reshape(X_train_pca, (len(X_train),eventWindow,num_components))
X_test_pca = np.reshape(X_test_pca,(len(X_test),eventWindow,num_components))

print "Shape of X_train_pca:", X_train_pca.shape
print "Shape of X_test_pca:", X_test_pca.shape

#X_train_pca = np.copy(X_train)
#X_test_pca = np.copy(X_test)

buffer = 1.5 #meaning we are going to have 1.5 timess the no. of neg samples than positive samples
ind_train_neg = np.where(Y_train[:,0]==0)[0].astype(int)
pos_train_samples = np.where(Y_train[:,0] == 1)[0].astype(int)
ind_train_neg = shuffle(ind_train_neg)
X_train_balanced = np.delete(X_train,ind_train_neg[len(pos_train_samples)*buffer:],0)

X_train_pca_balanced = np.delete(X_train_pca,ind_train_neg[len(pos_train_samples)*buffer:],0)

Y_train_balanced = np.delete(Y_train,ind_train_neg[len(pos_train_samples)*buffer:],0)

######################################################
######################################################
ind_test_neg = np.where(Y_test[:,0]==0)[0].astype(int)
pos_test_samples = np.where(Y_test[:,0]==1)[0].astype(int)
ind_test_neg = shuffle(ind_test_neg)
X_test_balanced = np.delete(X_test,ind_test_neg[len(pos_test_samples)*buffer:],0)

X_test_pca_balanced = np.delete(X_test_pca,ind_test_neg[len(pos_test_samples)*buffer:],0)

Y_test_balanced = np.delete(Y_test,ind_test_neg[len(pos_test_samples)*buffer:],0)

#scaling data between 0 and 1
X_train_pca_balanced_norm, X_test_pca_balanced_norm = getnorm(X_train_pca_balanced, X_test_pca_balanced,eventWindow,num_components)
X_train_balanced_norm, X_test_balanced_norm = getnorm(X_train_balanced, X_test_balanced, eventWindow,len(columns_in_data))

#shuffling
X_train_balanced, X_train_balanced_norm, X_train_pca_balanced, X_train_pca_balanced_norm, Y_train_balanced = shuffle(X_train_balanced, X_train_balanced_norm, X_train_pca_balanced, X_train_pca_balanced_norm, Y_train_balanced, random_state = 15)
X_test_balanced, X_test_balanced_norm, X_test_pca_balanced, X_test_pca_balanced_norm, Y_test_balanced = shuffle(X_test_balanced, X_test_balanced_norm, X_test_pca_balanced, X_test_pca_balanced_norm, Y_test_balanced, random_state = 5)


print "Shape of X_train_pca_balanced:",X_train_pca_balanced.shape
print "Shape of X_test_pca_balanced:",X_test_pca_balanced.shape

print "Shape of X_train_balanced:",X_train_balanced.shape
print "Shape of X_test_balanced:", X_test_balanced.shape
print "Shape of Y_train_balanced:", Y_train_balanced.shape
print "Shape of Y_test_balanced:",Y_test_balanced.shape

np.save(savepath_balanced+'X_train_20pca_balanced.npy',X_train_pca_balanced)
np.save(savepath_balanced+'X_test_20pca_balanced.npy',X_test_pca_balanced)
np.save(savepath_balanced+'X_train_20pca_balanced_norm.npy',X_train_pca_balanced_norm)
np.save(savepath_balanced+'X_test_20pca_balanced_norm.npy',X_test_pca_balanced_norm)
np.save(savepath_balanced+'Y_train_balanced.npy',Y_train_balanced)
np.save(savepath_balanced+'Y_test_balanced.npy',Y_test_balanced)
np.save(savepath_balanced+'X_train_balanced.npy',X_train_balanced)
np.save(savepath_balanced+'X_test_balanced.npy',X_test_balanced)
np.save(savepath_balanced+'X_train_balanced_norm.npy',X_train_balanced_norm)
np.save(savepath_balanced+'X_test_balanced_norm.npy',X_test_balanced_norm)


np.save(savepath+'X_train.npy', np.array(X_train))
np.save(savepath+'X_test.npy', np.array(X_test))
np.save(savepath+'Y_train.npy', np.array(Y_train))
np.save(savepath+'Y_test.npy', np.array(Y_test))

        






           

           



           


           

