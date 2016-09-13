import numpy as np
import os
import math
import matplotlib.pyplot as plt


num_landmark = 68
overhead = 3
sourcepath = "../processed/face_points"
savepath_folder = "../processed/histograms/"

#done = ["1458337615622674Feature","1459371665947140Feature","1460673068694649Feature","1461625577021029Feature","1462221806055660Feature","1462573185056370Feature","1463696819302037Feature"," 1464815752450717Feature","1458587118485511Feature","1459973682141964Feature","1460674041073330Feature","1461777269281809Feature","1462385517295671Feature","1462914895298967Feature","1463778374146728Feature"," 1464818081465795Feature","1458588237967448Feature","1459980768823544Feature","1461279139287978Feature","1461859557334768Feature","1462386868768725Feature","1462916449342755Feature","1463780274469495Feature"," 1464908651597426Feature","1458770010035679Feature","1460408240234532Feature","1461348328451936Feature","1461861353068630Feature","1462388981453229Feature","1462921325694628Feature","1463783394845512Feature"," 1465237240420997Feature","1459196537668690Feature","1460409533841827Feature","1461619357481758Feature","1461880518868044Feature","1462468904871091Feature","1463077101808539Feature","1464384689390932Feature"," 1465240573996308Feature","1459199120205582Feature","1460590425927638Feature","1461621235238498Feature","1461967704454056Feature","1462554563684426Feature","1463178417554658Feature"]
#done = [x.strip() for x in done]

duoface = ["1458254694982667Feature"]

if not os.path.isdir(savepath_folder):
	os.makedirs(savepath_folder)

#--------------------------------------------------------------------------------------------#
def horizontalHist(_F1, _F2):
	h_bins = [-4,-2,0,2,4]
	horizontal_dif = np.zeros(_F1.shape[0])
	for i in range(0,num_landmark):
		horizontal_dif[i] = _F2[i,0]-_F1[i,0]
	hist = list(np.histogram(horizontal_dif, h_bins, density=False))
	return [x/float(num_landmark) for x in hist]
#--------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------#
def angularHist(_F1, _F2):
	bins_coef = [0.0, 0.5, 1.0, 1.5, 2.0]
	p = math.pi
	a_bins = [x*p for x in bins_coef]
	ang_val = np.zeros(_F1.shape[0])
	vect = np.zeros(_F1.shape[0])
	for i in range(0,num_landmark):
		ang_val[i] =  math.pi + math.atan2((_F2[i,1]-_F1[i,1]), (_F2[i,0]-_F1[i,0]))
		vect[i] = math.hypot((_F2[i,0]-_F1[i,0]),(_F2[i,1]-_F1[i,1]))
	ahist = np.histogram(ang_val, a_bins, density=False)
	return [x/float(num_landmark) for x in ahist]
#--------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------#
def meanMotion(_F1, _F2):
	totalMotion = 0.0
	_num = _F1.shape[0]
	for i in range(_num):
		totalMotion+=math.hypot(_F2[i,0],_F2[i,1])-math.hypot(_F1[i,0],_F1[i,1])
	return totalMotion/_num
#--------------------------------------------------------------------------------------------#


print "Histogram creation started"

dirlist = os.listdir(sourcepath)
for dir in dirlist:
	#print dir
	if os.path.isdir(sourcepath + "/" + dir) and (dir[-7:] == "Feature") and (dir in duoface):
		print "Creating histograms for %s" %dir
		path = sourcepath + '/'+ dir + '/' 
		savedpath = savepath_folder + dir + '_hist/'
		if not os.path.isdir(savedpath[:-1]):
			os.makedirs(savedpath[:-1])
		
		files = sorted(os.listdir(path))
		files_pts = [i for i in files if i.endswith('.pts')]
		length = len(files_pts)
		
		for j in range (length-1):
			with open(path+files_pts[j]) as f:
		    		lines = f.readlines()
			F1 = np.zeros((num_landmark,2))
			for i in range(overhead,len(lines)-1):
				temp = lines[i].split(' ')
				F1[i-overhead,0] = float(temp[0])
				F1[i-overhead,1] = float(temp[1])

			with open(path+files_pts[j+1]) as f:
		    		lines = f.readlines()
			F2 = np.zeros((num_landmark,2))
			for i in range(overhead,len(lines)-1):
				temp = lines[i].split(' ')
				F2[i-overhead,0] = float(temp[0])
				F2[i-overhead,1] = float(temp[1])
			#print "F1: ",F1.shape
			#mP1 = np.mean(F1)
			#print "mP1: ",mP1	
			
			hist = np.reshape(horizontalHist(F1,F2)[0], (1,4))
			anghist = np.reshape(angularHist(F1,F2)[0], (1,4))
			mM = np.reshape(meanMotion(F1,F2), (1,1))

			#print "mM: ",mM
			fea = np.concatenate((hist,anghist), axis=1)
			fea = np.concatenate((fea,mM),axis=1)
			feat = ""
			for i in range(fea.shape[1]):
				feat += str(fea[0,i]) + " "			
			#print "feat: ",feat	
			test = feat.split()
			#print len(test)
			#print test
			testFloat = map(float, test)
			#print testFloat
			file = open(savedpath+files_pts[j+1][:-4] + '_hist.txt','w')
			file.write(feat)
			file.close()

print "Done creating histograms. Merging files..."
#merging all the txt files and creating a csv

import pandas as pd

path = savepath_folder

dirs = os.listdir(path)
for dir in dirs:
    files = os.listdir(path+dir+'/')
    print "Total no. of files in folder %s : %d" %(dir,len(files))
    histogram = []
    count = 0
    for fileName in files:
        if '.txt' in fileName:
            with open(path+dir+'/'+fileName,'r') as txtfile:
                timestamp = fileName[:fileName.index('_')]
                dummy = [timestamp]
                dummy.extend(txtfile.readline().split(' ')[:-1]) #to ignore end '\n' character
                histogram.append(dummy)
                count+=1
            txtfile.close()
    columns = ['timestamp','binX1','binX2','binX3','binX4','binTheta1','binTheta2','binTheta3','binTheta4','Mean']
    histogram_df = pd.DataFrame(histogram,columns=columns)
    histogram_df = histogram_df.set_index(['timestamp'])
    histogram_df.to_csv(path+dir[:dir.index('Feature')]+'_faceHist.csv')
    print "Processed %d files in folder %s " %(count,dir)

    os.system('rm -rf '+ path+dir)  #deleting the created text files




