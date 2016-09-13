# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:31:46 2016

@author: vijay
"""

import csv
import numpy as np
import matplotlib.pyplot as py
import savitzky_golay
import pandas as pd

fileName = 'gps.csv'
saveFileName = "gps_processed.csv"


"""parameters"""
avgWindow = 3
threshold =40  #setting a threshold distance of 40 meters
threshold2 = 15 #setting a second threshold for 15 meters
diffThreshold = 0.4
sDW = 3 #second diff threshold window to find areas where first diff is big enough (maxima or minima)


"""variables""" #assigned outside to monitor
dataPerSession = {}    
runAvg = {}
avgDiff = {}
yhat = {}
secondDiff = {}


def findHistIntersect(k,arr1,sgn,arr2):
    if(max(abs(np.array((arr1[k-sDW:k+sDW])))) > diffThreshold):
        return sgn
    else:
        return -1 #meaning he has stopped at an intersection


def readData(fileName):
    data =[]
    with open(fileName,'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader,None) #skip the header
        for row in reader:
            data.append([row[0],int(row[1]),float(row[2])])
        data = np.array(data)
    csvfile.close()
    return data



def dataSegregating(data):

    
    totalSessions = int(max(data[:,1].astype(int)))
    print totalSessions
    for i in range(totalSessions):  #sessionID
        dataPerSession[i+1] = np.array([data[k,2] for k in [j for j in range(len(data[:,1])) if data[j,1]==str(i+1)]]).astype(float)
    return dataPerSession

def dataProcessing(dataPerSession):
    totalSessions = len(dataPerSession)
    for i in range(totalSessions):
    #for i in range(3):
        gpsPerSession = dataPerSession[i+1]
        if len(gpsPerSession) == 0: #some sessions do not exist
            continue
        runAvg[i] = [0]
        avgDiff[i] = []
    
        for j in range(1,len(gpsPerSession)):
            diff = (gpsPerSession[j] - gpsPerSession[j-1])
            runAvg[i].append(diff)
            meanDiff = np.mean(runAvg[i][max(0,j-avgWindow):])                    
            avgDiff[i].append(meanDiff)
            
        yhat[i] = savitzky_golay.savitzky_golay(np.array(avgDiff[i]), 9, 4) # smoothing using window size 19, polynomial order 3
        secondDiff[i] = [0]*sDW
        
        for k in range(sDW,len(yhat[i]-sDW)):    
            if gpsPerSession[k] < threshold:
                sgn = 1*np.sign(yhat[i][k]-yhat[i][k-1])
                histIntersect = findHistIntersect(k,yhat[i],sgn,secondDiff[i])
                
                if gpsPerSession[k] < threshold2:
                    secondDiff[i].append(2*histIntersect)
                else:
                    secondDiff[i].append(1*histIntersect)
            else:
                secondDiff[i].append(0)
        secondDiff[i].append(0)
        
        #plotData(i,avgDiff[i],yhat[i],secondDiff[i])   #uncomment if you want to plot the data
    mergeData(secondDiff,data)
    #adding time converted stamp
    gpsData = pd.read_csv(saveFileName)
    gpsData = gpsData.set_index(['timestamp'])
    gpsData["timestamp_converted"] = pd.to_datetime(gpsData.index, unit='us')
    gpsData.to_csv(saveFileName)
    return secondDiff
    
    
    
def plotData(i,arr4,arr5,arr6):
    """plotting"""
    py.figure(i)
    py.plot(arr4) #avgDiff[i]
    py.plot(arr5) #yhat[i]
    py.plot(arr6) #secondDiff[i]



def mergeData(secondDiff,data):
    """merging with initial list"""
    finalList = []
    for indiList in secondDiff.values():
        finalList += indiList
    finalList=np.array(finalList).astype(str)
    data = np.insert(data,3,finalList,axis=1)
    data = np.insert(data,0,["timestamp","sessionID","intersectionDist","intersectionDirection"],axis=0)
    print data
    np.savetxt(saveFileName,data,fmt='%s',delimiter=",") #saving to csv


if __name__ == '__main__':
    data = readData(fileName)
    dataPerSession = dataSegregating(data)
    secondDiff = dataProcessing(dataPerSession)