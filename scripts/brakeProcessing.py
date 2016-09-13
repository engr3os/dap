# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:28:25 2016

@author: vijay
@motivation: tzhou
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = "../tripCsvFiles/"

def findlocalMax(x, W, tr):
    if W % 2 == 0:
        W += 1
    hw = int(W/2)
    N = len(x)
    
    lm = []
    i = hw
    while i < N-hw:
        if x[i] == np.amax(x[i-hw: i+hw+1]) and x[i] > tr:
            lm.append(i)
            i+=hw
        else:
            i+=1
    
    return np.array(lm)

#this function is to remove small hills in brake data
def reprocessZeroCross(brakePressure, zeroCros, thres, window):  
    zeroCrosRefined = []
    for i in zeroCros:
        if max(brakePressure[i:min(i+window,len(brakePressure)-1)]) > thres:
            zeroCrosRefined.append(i)
    return zeroCrosRefined
    

def processBrakeData(brakePressure, slopeTr, show=0,fig=1): #give brakePressure as list or numpy array
    brakePressure = np.array(brakePressure)
    N = len(brakePressure)
    
    nonzero = np.nonzero(brakePressure > 0.01)[0]

    # get slope
    brakePressureSlope = np.correlate(brakePressure, [-1,1], mode='same')
    uprise = np.nonzero(brakePressureSlope > slopeTr)[0]
    downrise = np.nonzero(brakePressureSlope < -slopeTr)[0]

    # find local max on brake pressure
    localMax = findlocalMax(brakePressureSlope, W=31, tr=slopeTr)

    # get zero crossing
    shiftedbrakePressure = brakePressure - 0.001
    zeroCros = []
    for i in range(N-1):
        if shiftedbrakePressure[i+1] >= 0 and shiftedbrakePressure[i] < 0:
            zeroCros.append(i)
    
    thres = 0.2
    zeroCrosRefined = reprocessZeroCross(brakePressure, zeroCros, thres, window = 20) #20 implies 2 sec window.
    print "Number of zero crossing: %i" % len(zeroCrosRefined)
    zeroCrosRefined = np.array(zeroCrosRefined)
    sumThreshold =1 #this is the sum of the total brake pressure over the window interval to accomodate small peaks or merge two braking events    
    
    
    """extracting only the brake events with noBrakeWindow of zero brake pressure before the event for easier training"""
    noBrakeWindow = 50 #50 implies 5 seconds
    sparseZeroCros = [zeroCrosRefined[0]]
    brakePressureClipped = [i if i > thres else 0 for i in brakePressure]
    for i in range(1,len(zeroCrosRefined)):
        if zeroCrosRefined[i] - zeroCrosRefined[i-1] > noBrakeWindow and sum(brakePressureClipped[max(0,zeroCrosRefined[i]-noBrakeWindow):zeroCrosRefined[i]]) < sumThreshold:
            sparseZeroCros.append(zeroCrosRefined[i])
    print "Number of sparse zero crossing: %i" % len(sparseZeroCros)
    sparseZeroCros = np.array(sparseZeroCros)

    """extracting events where there is no braking for a specified window time"""
    emptyPoints = []
    pointer = 0
    while pointer < N-noBrakeWindow:
        if sum(brakePressure[pointer:min(N,pointer+2*noBrakeWindow)]) < sumThreshold:   #using 2*noBrakeWindow since I wanted to avoid having a braking activity approaching in 5 secs after a 5 sec window
            pointer += noBrakeWindow
            emptyPoints.append(pointer) #the end of the empty window is appended
        else:
            pointer += noBrakeWindow
    print "Number of empty windows found: %d" %len(emptyPoints)
        
    
    
    # plot them
    if show:
        #plt.subplot(10,1,1)
        """
        plt.figure(fig)
        plt.plot(brakePressureSlope, 'g-')
        plt.title('brake pressure slope')
        
        plt.subplot(10,1,2)
        plt.plot(brakePressure, 'g-')
        plt.plot(uprise, brakePressure[uprise], 'r*')
        plt.title('uprise edge')

        plt.subplot(10,1,3)
        plt.plot(brakePressure, 'g-')
        plt.plot(downrise, brakePressure[downrise], 'r*')
        plt.title('downrise edge')

        plt.subplot(10,1,4)
        plt.plot(brakePressure, 'g-')
        plt.plot(nonzero, brakePressure[nonzero], 'r*')
        plt.title('non zero value')
        
        plt.subplot(10,1,5)
        plt.plot(brakePressureSlope, 'g-')
        plt.plot(localMax, brakePressureSlope[localMax], 'r*')
        plt.title('brake pressure slope local max')
        
        #plt.subplot(10,1,6)
        plt.figure(fig+1)        
        plt.plot(brakePressure, 'g-')
        plt.plot(localMax, brakePressure[localMax], 'r*', markersize = 10)
        plt.title('brake pressure local max')
        """
        plt.figure(fig+2)        
        plt.plot(brakePressure, 'g-')
        plt.plot(sparseZeroCros, brakePressure[sparseZeroCros], 'b*', markersize=10)
        plt.title('brake pressure sparse zero crossing')
        
        plt.figure(fig+3)        
        plt.plot(brakePressure, 'g-')
        plt.plot(emptyPoints, brakePressure[emptyPoints], 'g*', markersize=10)
        plt.title('brake pressure empty points')
        
        """
        #plt.subplot(10,1,7)
        plt.figure(fig+4)        
        plt.plot(brakePressure, 'g-')
        plt.plot(zeroCrosRefined, brakePressure[zeroCrosRefined], 'r*', markersize = 10)
        plt.title('brake pressure zero crossing')
        """        
        

        plt.show()

    # format label
    ynonzero = [1 if i in nonzero else 0 for i in range(N) ]
    yuprise = [1 if i in uprise else 0 for i in range(N) ]
    ydownrise = [1 if i in downrise else 0 for i in range(N) ]
    ySlopeLocalMax = [1 if i in localMax else 0 for i in range(N) ]
    yzeroCrossing = [1 if i in zeroCrosRefined else 0 for i in range(N) ]
    ySparseZeroCrossing = [1 if i in sparseZeroCros else 0 for i in range(N) ]
    yEmptyPoints = [1 if i in emptyPoints else 0 for i in range(N) ]
    
    
    
    return np.array(yEmptyPoints), np.array(yzeroCrossing), np.array(ySparseZeroCrossing)
    
def resample(df):
    RESAMPLE = '100L'
    df = df[df.index!=df.index[0]]
    df = df[df.index!=df.index[-1]]
    df = df.resample(RESAMPLE).first()
    #df = df.interpolate()
    
    return df['pbrk'].values.tolist(),df
    
    
if __name__ == '__main__':
    file = '1463696819302037_features_agg.csv'
    destfile = pd.read_csv(path+file, skiprows=(1,2))
    destfile = destfile.set_index(destfile['timestamp'])
    folder = file[:file.index('_')]
    brakePressureSampled = destfile['pbrk'].values.tolist()
    #brakecsv = pd.read_csv(path + file)
    #brakecsv = brakecsv.set_index(pd.DatetimeIndex(brakecsv['timestamp']))
    #brakecsv.index = pd.to_datetime(brakecsv.index, unit='us')
    #brakePressureSampled, brakecsv = resample(brakecsv)
    
    #brakePressureSampled = brakecsv.resample('100L').first()
    #brakePressure = brakecsv['pbrk'].values.tolist()
    
    #ySlopeLocalMax, yzeroCrossing = processBrakeData(brakePressure, slopeTr = 0.05, show = 1, fig =1)
    yEmptyPointsSampled, yZeroCrossingSampled, ySparseZeroCrossingSampled = processBrakeData(brakePressureSampled, slopeTr = 0.05, show = 1,fig =4)
    #brakecsv['zeroCross'] = ySparseZeroCrossingSampled    
    #print brakecsv

    
 
