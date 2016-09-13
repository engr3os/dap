# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:57:49 2016

@author: vijay
"""
savepath = '../data_for_network_balanced/'
import numpy as np
from sklearn import svm


def calcAccuracy(pred,act):
    hit = 0
    count = 0
    for i in range(len(pred)):
        if int(pred[i])==int(act[i]):
            hit+=1
            count+=1
        else:
            count+=1
    print "hit, count, accuracy: ", (hit,count,1.0*hit/count)


X_train = np.load(savepath + 'X_train_20pca_balanced.npy')
X_test = np.load(savepath + 'X_test_20pca_balanced.npy')
Y_train = np.load(savepath+ 'Y_train_balanced.npy')
Y_test = np.load(savepath + 'Y_test_balanced.npy')
totalSamples = Y_train.shape[0]*Y_train.shape[1] + Y_test.shape[0]*Y_test.shape[1]
print "Total no. of samples ", totalSamples
ratio = 1.0*(sum(Y_train) + sum(Y_test))/totalSamples
print "Ratio of positive samples in set: ", ratio
featureSize = X_train.shape[2]

#X_train = X_train[:,0]
print "Shape of X_train before reshaping: ", X_train.shape
X_train =  np.reshape(X_train,[-1,featureSize])
#X_test = X_test[:,0]
X_test = np.reshape(X_test,[-1,featureSize])

#Y_train = Y_train[:,0]
Y_train = np.reshape(Y_train,-1)
#Y_train = np.mean(Y_train,axis=1)
print max(Y_train)
#Y_test = Y_test[:,0]
Y_test = np.reshape(Y_test,-1)
#Y_test = np.mean(Y_test,axis =1)
print max(Y_test)
print "Size of input: ", X_train.shape
c=0.1
print "Value of C:", c
clf = svm.SVC(C=c)
#clf = svm.SVC(class_weight="auto")
clf.fit(X_train,Y_train)
print "Done training"
output_train = clf.predict(X_train)
assert len(output_train) == len(Y_train)
print "Training Accuracy: "
calcAccuracy(output_train,Y_train)


print "Testing started"
output = clf.predict(X_test)
assert len(output)==len(Y_test)
calcAccuracy(output,Y_test)



Y_test_save = np.reshape(Y_test,[-1,50])
output_save = np.reshape(output,[-1,50])
np.savetxt(savepath+"y_labels.txt",Y_test_save,fmt='%i')
np.savetxt(savepath+"pred_labels.txt",output_save,fmt='%i')

