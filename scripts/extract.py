#! /usr/bin/env python
# author : Jun Liu 2016

import os
import pickle
import pandas as pd
import numpy as np
import csv

PATH = "../pickleFiles/"
SAVEPATH = "../tripCsvFiles/"
LIST = os.listdir(PATH)
sessionIndexFile = "sessionIndex.csv"
gpsFile = "gps_processed.csv"
facefolderPath = '../processed/histograms/'
handFolderPath = '../processed/handfeatures_processed/'

if not os.path.isdir(SAVEPATH[:-1]):
    os.makedirs(SAVEPATH[:-1])


FILES = []

for file in LIST:
    if '_extr.pkl' in file and 'lock' not in file:
        FILES.append(file)
        
print "Total no. of files: %d" %len(FILES)


RESAMPLE = '100L' #100 milliseconds frequency
timediff = 0.1 #set this according to the above parameter



def extractSessionIndexAndGpsFile(sessionIndexFile, gpsFile):
    sessionIndex = pd.read_csv(sessionIndexFile)
    gpsData = pd.read_csv(gpsFile)
    gpsData = gpsData.set_index(['timestamp'])
    gpsData.index = pd.to_datetime(gpsData.index, unit='us')
    return sessionIndex,gpsData

def extractFaceAndHandData(folder,facefolderPath,handFolderPath):
    if os.path.isfile(facefolderPath+folder+'_faceHist.csv'):
        faceData_df = pd.read_csv(facefolderPath+folder+'_faceHist.csv')
        faceData_df = faceData_df.sort_values(by='timestamp',ascending=1)
        faceData_df = faceData_df.set_index(['timestamp'])
        faceData_df.index = pd.to_datetime(faceData_df.index, unit='us')
    else:
        print "Couldn't find face data for folder %s" %folder
        faceData_df = pd.DataFrame([])  #return empty data frame to be checked in parent function
    
    if os.path.isfile(handFolderPath+folder+'/feature/txtHandsOnWheel.txt'):
        handData_df = pd.read_csv(handFolderPath+folder+'/feature/txtHandsOnWheel.txt',sep=" ",header=None)
        handData_df = handData_df.drop(handData_df.columns[[-1]],axis=1)
        
        handData_df.columns=['timestamp','numHandsOnWheel','gripClusterID','left_x0','left_y0','left_d','left_theta','left_onOff',
                             'left_isMoving','left_movingD','left_movingTheta','right_x0','right_y0','right_d','right_theta',
                             'right_onOff','right_isMoving','right_movingD','right_movingTheta','dBeweenHands','angleBetweenHands']
        #the list of this columns can be found at sftp://10.0.0.165/home/tzhou/Workspace/interns/BrakePrediction/src/HandCamReader_vijay.py
       
        handData_df['timestamp'] = handData_df['timestamp'].multiply(1e5) #to compensate for the division by 1e-5 in the above commented file        
        handData_df = handData_df.sort_values(by='timestamp',ascending=1)        
        handData_df = handData_df.set_index(['timestamp'])        
        handData_df.index = pd.to_datetime(handData_df.index, unit='us')
    else:
        print "Couldn't find hand data for folder %s" %folder
        handData_df = pd.DataFrame([])  #return empty data frame to be checked in parent function
    return faceData_df,handData_df
    


def resample(df):
    df = df[df.index!=df.index[0]]
    df = df[df.index!=df.index[-1]]
    df = df.resample(RESAMPLE).mean()
    df = df.interpolate()
    return df

def object_resample(object_df):
    def custom(array):
        array=array[array>0]
        return np.min(array)
    # sync object_log
    left_lb = -4.5
    left_rb = -1.5
    right_lb = 1.5
    right_rb = 4.5
    object_df = object_df.reset_index()
    left = object_df[['posX','velX']].copy()
    mid = left.copy()
    right = left.copy()
    left.columns = [u'lposX', u'lvelX']
    mid.columns = [u'mposX', u'mvelX']
    right.columns = [u'rposX', u'rvelX']
    left.loc[object_df['posY']>left_rb]=0
    left.loc[object_df['posY']<left_lb]=0
    mid.loc[object_df['posY']<left_rb]=0
    mid.loc[object_df['posY']>right_lb]=0
    right.loc[object_df['posY']<right_lb]=0
    right.loc[object_df['posY']>right_rb]=0
    object_df=pd.concat([object_df['timestamp'],left,mid,right],axis=1)
    object_df = object_df.set_index(['timestamp'])
    object_df = object_df.resample(RESAMPLE).agg({'lposX':custom,
                                                  'lvelX':np.mean,
                                                  'mposX':custom,
                                                  'mvelX':np.mean,
                                                  'rposX':custom,
                                                  'rvelX':np.mean})
    object_df.fillna(0,inplace=True)
    #object_df = object_df.interpolate()
    return object_df

def lane_resample(lane_df):
    # sync lane_log
    lane_df = lane_df[['leftlane_valid','leftlane_offset','leftlane_curvature','leftlane_boundarytype',
                       'rightlane_valid','rightlane_offset','rightlane_curvature','rightlane_boundarytype']].copy()
    lane_df = lane_df.resample(RESAMPLE).first()
    lane_df.fillna(0,inplace=True)
    return lane_df

def gpsDist_resample(gpsDist_df):
    gpsDist_df = gpsDist_df[gpsDist_df.index!=gpsDist_df.index[0]] #removing the anomolous 1970 errors
    gpsDist_df = gpsDist_df[gpsDist_df.index!=gpsDist_df.index[-1]]
    
    gpsDist_df = gpsDist_df.resample(RESAMPLE).first()
    gpsDist_df['intersectionDist'] = gpsDist_df['intersectionDist'].interpolate(method='index')
    #cols = ['sessionID','intersectionDirection']
    #gpsDist_df[cols]= gpsDist_df[cols].ffill()
    gpsDist_df['sessionID'] = gpsDist_df['sessionID'].ffill()
    gpsDist_df['intersectionDirection'] = gpsDist_df['intersectionDirection'].ffill().round(2)
    gpsDist_df.fillna(0,inplace=True)
    return gpsDist_df

def face_hand_resample(df):
    if df.size==0:
        return df
    #df = df[df.index!=df.index[0]]
    #df = df[df.index!=df.index[-1]]
    df = df.resample(RESAMPLE).first()
    df = df.ffill()  #forward filling
    return df
    
def gps_resample(gps_df):
    gps_df.fillna(0,inplace=True)
    gps_df = gps_df[gps_df['latitude']!=0]
    gps_df = gps_df[gps_df['longitude']!=0]
    gps_df = gps_df[['latitude', 'longitude', 'elevation']]
    gps_df = gps_df.resample(RESAMPLE).first()
    gps_df = gps_df.interpolate()
    print gps_df.shape
    return gps_df

def add_head(path, filename, head0, head1):
    # add head
    fin = open(os.path.join(path,filename))
    incsv = csv.reader(fin)
    fout = open(os.path.join(path,filename[:-4]+'_agg.csv'),'w')
    outcsv = csv.writer(fout)
    count = 0
    for row in incsv:
        outcsv.writerow(row)
        if count == 0:
            outcsv.writerow(head0)
            outcsv.writerow(head1)
        count +=1
    fin.close()
    fout.close()
    os.remove(os.path.join(path,filename))


def extr(File, path):
    filename = path+File
    folder = File[:File.index('_')]
    head0=[]
    head1=[]
    print filename
    print path
    
    sessionIndex,gpsData = extractSessionIndexAndGpsFile(sessionIndexFile,gpsFile) #parsing sessionMap and GpsDist files
    
    data_face,data_hand = extractFaceAndHandData(folder,facefolderPath,handFolderPath)
    
    #getting session number from folder name
    sessionNumber = sessionIndex.session_id[sessionIndex.timestamp.astype(str)==folder].values
    if sessionNumber.size == 0:
        print "Couldn't find session number for folder %s" %folder
        return
    else:
        sessionNumber = sessionNumber[0]
        print "Session ID is %s" %sessionNumber


    #get the intersection direction from gps_processed
    gpsDist = gpsData[gpsData.sessionID == sessionNumber]
    #print gpsDist
    #print type(gpsDist)
    if gpsDist.values ==[]:
        print "Couldn't find data for session %d and folder %s " %sessionNumber, folder
        return
    
    with open(filename,'rb') as f:
        print 'processing %s' % filename
        data = pickle.load(f)
        data_str_angle = data[0]['data']
        data_pbrk = data[1]['data']
        data_hv_accp = data[2]['data']
        data_yaw_rate = data[4]['data']
        data_gps = data[6]['data']
        data_obd = data[7]['data']
        data_object = data[8]['data']
        data_lane = data[9]['data']
        data_gps_inter_dist = gpsDist
        data_accel = data_obd.speed.diff().astype(float).divide(timediff)
        data_accel.fillna(0,inplace=True)  #Removing NaNs from acceleration
        data_accel = data_accel.to_frame(name='acceleration')
        #print type(data_accel)
        #print data_accel
        
        """write braking data to csv before resampling for later processing of target"""
        data_pbrk.to_csv(SAVEPATH+folder+'_unsampled_brake.csv')

        data_str_angle = resample(data_str_angle)
        data_pbrk = resample(data_pbrk)
        data_hv_accp = resample(data_hv_accp)
        data_yaw_rate = resample(data_yaw_rate)
        data_gps = gps_resample(data_gps)
        data_obd = resample(data_obd)
        data_accel = resample(data_accel)
        data_object = object_resample(data_object)
        data_lane = lane_resample(data_lane)
        data_gps_inter_dist = gpsDist_resample(data_gps_inter_dist)
        data_face = face_hand_resample(data_face)
        data_hand = face_hand_resample(data_hand)
        #print data_accel        
        #print data_gps_inter_dist

        data_ret = data_obd.merge(data_gps,how='inner',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_accel, how='left',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_gps_inter_dist, how='left',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_object,how='left',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_lane,how='left',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_str_angle,how='left',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_pbrk,how='left',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_hv_accp,how='left',left_index=True,right_index=True)
        data_ret = data_ret.merge(data_yaw_rate,how='left',left_index=True,right_index=True)
        if data_hand.size!=0:
            data_ret = data_ret.merge(data_hand,how='right',left_index=True,right_index=True) #giving second priority to hand data
        if data_face.size!=0:
            data_ret = data_ret.merge(data_face,how='right',left_index=True,right_index=True) #giving first priority to face data
        
        data_ret.fillna(0,inplace=True)
        bins = [i+0.5 for i in range(300)]   #setting the max speed, mposX, lposX and rposX to 300
        data_ret['speed']=pd.np.digitize(data_ret['speed'], bins = bins)
        data_ret['mposX']=pd.np.digitize(data_ret['mposX'], bins = bins)
        data_ret['lposX']=pd.np.digitize(data_ret['lposX'], bins = bins)
        data_ret['rposX']=pd.np.digitize(data_ret['rposX'], bins = bins)
        data_ret['engine_rpm']=data_ret['engine_rpm'].astype(int)        
        
        # add head
        col = ['timestamp']
        for i in data_ret.columns:
            col.append(i)
        head = pd.DataFrame(columns=col)
        head0 = ['datetime']
        head1 = ['T']
        for i in data_ret.columns:
            head0.append('float')
            head1.append('')
        head.loc[0]=head0
        head.loc[1]=head1
        head= head.set_index(['timestamp'])
        
        features = folder + '_features.csv'
        
        data_ret.to_csv(
            os.path.join(SAVEPATH,features))
    # add head
    add_head(SAVEPATH,features,head0,head1)
    print "Finished processing folder %s" %folder

if __name__ == '__main__':
    for i in range(len(FILES)):
        try:
            extr(FILES[i],PATH)
        except:
            #print "Error processing folder: %s" %FILES[i][:FILES.index('_')]
            print "Error processing folder: %s" %FILES[i]
            continue

