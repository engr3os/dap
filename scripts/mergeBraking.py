# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:25:13 2016

@author: vijay
"""

import os
import pandas as pd
import brakeProcessing

destpath = '../tripCsvFiles/'
destfiles = []
filelist = os.listdir(destpath)


for file in filelist:
    if 'features_agg.csv' in file:
        destfiles.append(file)
print "Total no. of files found :%d" %len(destfiles)

if __name__ == '__main__':
    slopeTr = 0.05
    show = 0
    countTotalBrakeEvents = 0
    countTotalSparseBrakeEvents = 0
    countTotalEmptyWindows = 0
    for file in destfiles:
        destfile = pd.read_csv(destpath+file, skiprows=(1,2))
        destfile = destfile.set_index(destfile['timestamp'])
        folder = file[:file.index('_')]
        brakePressureSampled = destfile['pbrk'].values.tolist()
        print "Started processing folder %s" %folder    
        yEmptyPointsSampled, yzeroCrossingSampled, ySparseZeroCrossingSampled = brakeProcessing.processBrakeData(brakePressureSampled, slopeTr,show)
        
        countTotalBrakeEvents += sum(yzeroCrossingSampled)
        countTotalSparseBrakeEvents += sum(ySparseZeroCrossingSampled)
        countTotalEmptyWindows += sum(yEmptyPointsSampled)
        
        
        destfile['brakeEvents'] = yzeroCrossingSampled
        destfile['sparseBrakeEvents'] = ySparseZeroCrossingSampled
        destfile['emptyBrakeEvents'] = yEmptyPointsSampled
        print sum(destfile['sparseBrakeEvents'].values)        
        #print brakecsv
        if os.path.isfile(destpath+folder+'_features_brake_agg.csv'):
            os.system('rm ' + destpath+folder+'_features_brake_agg.csv')
        destfile.to_csv(destpath+folder+'_features_brake_agg.csv')
        print "Finished processing folder %s" %folder
    print "Total no. of brake events: %d" %countTotalBrakeEvents
    print "Total no. of sparse brake events: %d" %countTotalSparseBrakeEvents
    print "Total no. of empty brake events: %d" %countTotalEmptyWindows