"""
Note: this script is built only for metrics in binary classification
Creator: Vijay Ch
"""


import sys
import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score,f1_score
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_fscore_support


def processResults(resultLabel,testLabel,rawLabel,line_color = 'b',show=0):
	plt.close('all')

	def getplot(fp,fn,label_fp,label_fn):
		plt.plot(range(-len(fp),0),fp, color=line_color ,marker = 'x',label=label_fp)
		plt.plot(-len(fp),fp[0],marker='p',color='y',markersize=10)
		plt.plot(-1,fp[-1],marker='p',color='y',markersize=10)
		plt.plot(range(-len(fn),0),fn, color='r' ,marker = 'o',label=label_fn)
		plt.plot(-len(fn),fn[0],marker='p',color='y',markersize=10)
		plt.plot(-1,fn[-1],marker='p',color='y',markersize=10)
		plt.annotate('(%.3f)' % fp[0], xy = (-len(fp),fp[0]), textcoords = 'data')
		plt.annotate('(%.3f)' % fp[-1], xy = (-1,fp[-1]), textcoords = 'data')
		plt.annotate('(%.3f)' % fn[0], xy = (-len(fn),fn[0]), textcoords = 'data')
		plt.annotate('(%.3f)' % fn[-1], xy = (-1,fn[-1]), textcoords = 'data')
		plt.xlabel('Time to maneuver')
		plt.ylabel('Value')

	def runningmax(_label):
		
		length = len(_label)
		rmaxLabel = np.zeros(length)
		for i in range(1,length):
			c = Counter(_label[0:i])
			value, count = c.most_common()[0]
			
			rmaxLabel[i] = value
		rmaxLabel[0] = _label[0]
		return rmaxLabel
		#return int(sum(_label)>int(len(_label)/2))

	def get_confusion_matrix(y_hat,y_actual):
		TP = 0
		FP = 0
		TN = 0
		FN = 0

		for i in range(len(y_hat)): 
		    if y_actual[i]==y_hat[i]==1:
		       TP += 1
		for i in range(len(y_hat)): 
		    if y_actual[i]==0 and y_hat[i]==1:
		       FP += 1
		for i in range(len(y_hat)): 
		    if y_actual[i]==y_hat[i]==0:
		       TN += 1
		for i in range(len(y_hat)): 
		    if y_actual[i]==1 and y_hat[i]==0:
		       FN += 1

		#precisionP = 1.0*TP/(TP+FP)
		#recallP = 1.0*TP/(TP+FN)
		#precisionN = 1.0*TN/(TN+FN)
		#recallN = 1.0*TN/(TN+FP)

		return np.array([[TP, FN],[FP, TN]])

	def clean_y(y,pred_y,raw_pred_labels):
		sequenceLength = y.shape[1]
		y = np.reshape(y,-1).tolist()
		pred_y = np.reshape(pred_y,-1).tolist()
		raw_pred_labels = np.reshape(raw_pred_labels,-1).tolist()
		ind = [i for i in range(len(y)) if y[i]!=2]   #to remove the 2
		print "Length of ind: ", len(ind)
		if np.array(y).shape==np.array(pred_y).shape:  #meaning the actual label and the predicted label include the braking zones (otherwise only the actual label includes the braking zones)
			pred_y = [pred_y[i] for i in ind]
			tmp = []
			for i in ind:
				tmp.extend(raw_pred_labels[i*num_classes:(i+1)*num_classes])
			raw_pred_labels = tmp
		y = [y[i] for i in ind]
		y = np.reshape(np.array(y),[-1,sequenceLength])
		pred_y = np.reshape(np.array(pred_y),[-1,sequenceLength])
		raw_pred_labels = np.reshape(np.array(raw_pred_labels),[-1,sequenceLength*num_classes])
		return y,pred_y,raw_pred_labels

	def get_metrics(pred_y,y):
		pw_acc = []
		posNum = sum(y[:,0])
		negNum = len(y) - posNum
		print 'posNum: ', posNum
		print 'negNum: ', negNum
		tp,fp,tn,fn,precisionP,recallP,precisionN,recallN = [],[],[],[],[],[],[],[]
		for j in range(len(y[0])):
			conf_mat = get_confusion_matrix(pred_y[:,j],y[:,j])
			tp.append(conf_mat[0,0])
			fn.append(conf_mat[0,1])
			fp.append(conf_mat[1,0])
			tn.append(conf_mat[1,1])
			hit = tp[-1]+tn[-1]
			count = tp[-1]+tn[-1] + fp[-1]+fn[-1]
			print "time to action, piecewise accuracy, count_rm : (%.1f, %.4f, %d)" %(1.0*(pred_y.shape[1]-j-1)/10, 1.0*hit/count, count)
			pw_acc.append(1.0*hit/count)

		#print "tp:", tp
		#print "tn:", tn
		#print "fp:", fp
		#print "fn:", fn
	    
		for i in range(len(tp)):
			tmpP = precision_score(y[:,i],pred_y[:,i],average=None,pos_label=1)
			tmpR = recall_score(y[:,i],pred_y[:,i],average=None,pos_label=1)
			precisionP.append(tmpP[1])
			recallP.append(tmpR[1])
			precisionN.append(tmpP[0])
			recallN.append(tmpR[0])
			"""
			precisionP.append(1.0*tp[i]/(tp[i]+fp[i]))
			recallP.append(1.0*tp[i]/(tp[i]+fn[i]))
			precisionN.append(1.0*tn[i]/(tn[i]+fn[i]))
			recallN.append(1.0*tn[i]/(tn[i]+fp[i]))
			"""
		# print "precisionP:", precisionP
		# print "recallP:", recallP
		# print "precisionN:", precisionN
		# print "recallN:", recallN
		
		tpr = recallP
		fpr = recallN
		return tp,fp,tn,fn,precisionP,recallP,precisionN,recallN, tpr, fpr, pw_acc


	frame_index = 25  #set the index from the event of maneuver where you want to check precision, recall across various thresholds
	print "frame_index: ", frame_index

	num_classes = 2  
	global pred_y
	pred_y = np.loadtxt(resultLabel)
	print "pred_y[0]: ", pred_y[0] 

	global y
	global save_pred_y
 	global pred_label_tosave


	if 'npy' in testLabel:
		y = np.load(testLabel)
	elif 'txt' in testLabel:
		y = np.loadtxt(testLabel)
	else:
		y=testLabel
	print "y_test[0]: ", y[0]

	print "Sum of y :", sum([i for i in y.flatten() if i==0 or i==1])
	print "Sum of y only 0 :", sum([1 for i in y.flatten() if i==0])

	raw_pred_labels = np.loadtxt(rawLabel)
	eventWindow = y.shape[1]

	""" set this parameter if you want to save the predicted labels trip wise. 
	Note:: You need to have the ../tripCsvFiles folder in your directory to do this"""
	savePredLabelsTripWise = 1


	

	"""to remove the braking parts to check accuracy only when brake-pressure is zero (or the non braking zones)"""
	brakingZonesInY = 0
	if 2 in y:
		brakingZonesInY = 1
		save_y = np.copy(y)
		save_pred_y = np.copy(pred_y)
		y,pred_y,raw_pred_labels = clean_y(y,pred_y,raw_pred_labels)


	print "pred_y.shape: ",pred_y.shape
	print "y.shape: ",y.shape
	print "Raw labels shape: ", raw_pred_labels.shape

	"""get the metrics"""
	tp,fp,tn,fn,precisionP,recallP,precisionN,recallN, tpr, fpr, pw_acc = get_metrics(pred_y,y)

	"""get roc curve"""
	raw_pred_labels = raw_pred_labels[:,range(1,raw_pred_labels.shape[1],2)]   #extracting only the positive probabilities
	raw_one_prob = raw_pred_labels[:,-frame_index]  #Extracting the probability of 1 for that highest accuracy timestamp
	actual_prob = y[:,-frame_index]
	global false_positive_rate; global true_positive_rate
	false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(actual_prob, raw_one_prob,pos_label = 1)
	roc_auc = auc(false_positive_rate, true_positive_rate)


	"""reduce FP by using manual threshold"""
	global f1_grid
	precisionP_thres,recallP_thres,precisionN_thres,recallN_thres,f1_grid = [],[],[],[],[]
	thresholds = np.concatenate((np.arange(0,1,0.01),[1]))

	for thres in thresholds:
		pred_y_thres = np.copy(pred_y)#taking the last column
		for i in range(len(pred_y_thres)):
			#continue
			if pred_y_thres[i,-frame_index] == 1 and raw_pred_labels[i,-frame_index]<thres: #reducing number of FPs
				pred_y_thres[i,-frame_index] = 0
		
		tmp = precision_score(y[:,-frame_index], pred_y_thres[:,-frame_index], average=None,pos_label=1)
		precisionN_thres.append(tmp[0])
		precisionP_thres.append(tmp[1])
		tmp = recall_score(y[:,-frame_index], pred_y_thres[:,-frame_index], average=None,pos_label=1)
		recallN_thres.append(tmp[0])
		recallP_thres.append(tmp[1])
		tmp = []
		pred_y_thres = np.array(pred_y_thres)
		pred_y_thres[np.logical_and(pred_y_thres==1,np.array(raw_pred_labels)<thres)] = 0
		for j in range(eventWindow):
			tmp.append(f1_score(y[:,-j], pred_y_thres[:,-j],pos_label=1))
		f1_grid.append(tmp)
	f1_grid = np.array(f1_grid)
	print "shape of f1_grid", f1_grid.shape



	"""get metrics with manual threshold"""
	pred_y_thres = np.copy(pred_y)
	manual_thres = 0.60
	for i in range(len(pred_y_thres)):
		for j in range(len(pred_y_thres[0])):
			if pred_y_thres[i,j] == 1 and raw_pred_labels[i,j]<manual_thres: #reducing number of FPs
				pred_y_thres[i,j] = 0

	tp,fp,tn,fn,precisionP,recallP,precisionN,recallN, tpr, fpr, pw_acc_thres = get_metrics(pred_y_thres,y)





########plotting################

	fig9= plt.figure(9)
	plt.plot(range(-len(pw_acc),0),pw_acc, color=line_color , marker = 'o',label='rnn')
	plt.plot(-len(pw_acc),pw_acc[0],marker='p',color='y',markersize=10)
	plt.plot(-1,pw_acc[-1],marker='p',color='y',markersize=10)
	plt.axis([-50, 0, 0.5, 1])
	plt.annotate('(%.3f)' % pw_acc[0], xy = (-len(pw_acc),pw_acc[0]), textcoords = 'data')
	plt.annotate('(%.3f)' % pw_acc[-1], xy = (-1,pw_acc[-1]), textcoords = 'data')
	plt.grid(True)
	plt.title("Piecewise accuracy vs time to maneuver before thresholding")
	plt.xlabel('Time to maneuver')
	plt.ylabel('Accuracy')
	if show: plt.show()

	fig1 = plt.figure(1)
	plt.title('Receiver Operating Characteristic at time = '+ str(-1.0*frame_index/10)+ 'sec')
	plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'k--')
	plt.xlim([-0.1,1.1])
	plt.ylim([-0.1,1.1])
	#print "Number of thresholds: ", len(roc_thresholds)
	roc_thresholds = roc_thresholds.tolist()
	for i in range(0,len(roc_thresholds),25):
		plt.plot(false_positive_rate[i],true_positive_rate[i],marker='p',color='g',markersize=6)
		plt.annotate('(%.2f)' % roc_thresholds[i], xy = (false_positive_rate[i],true_positive_rate[i]), textcoords = 'data')
	plt.ylabel('True Positive Rate (Correct Detection)')
	plt.xlabel('False Positive Rate (False Alarm)')
	if show: plt.show()
	
	fig2= plt.figure(2)
	plt.plot(range(-len(pw_acc_thres),0),pw_acc_thres, color=line_color , marker = 'o',label='rnn')
	plt.plot(-len(pw_acc_thres),pw_acc_thres[0],marker='p',color='y',markersize=10)
	plt.plot(-1,pw_acc_thres[-1],marker='p',color='y',markersize=10)
	plt.axis([-50, 0, 0.5, 1])
	plt.annotate('(%.3f)' % pw_acc_thres[0], xy = (-len(pw_acc_thres),pw_acc_thres[0]), textcoords = 'data')
	plt.annotate('(%.3f)' % pw_acc_thres[-1], xy = (-1,pw_acc_thres[-1]), textcoords = 'data')
	plt.grid(True)
	plt.title("Piecewise accuracy vs time to maneuver at thres="+str(manual_thres))
	plt.xlabel('Time to maneuver')
	plt.ylabel('Accuracy')
	if show: plt.show()

	fig3= plt.figure(3)
	getplot(tp,tn,'tp','tn')
	#plt.axis([-50, 0, 0, 1])
	plt.grid(True)
	plt.title("Tp and Tn vs time to maneuver at thres="+str(manual_thres))
	plt.legend(bbox_to_anchor=(0.85, 0.7), loc=2, borderaxespad=0.)
	if show: plt.show()

	fig4= plt.figure(4)
	getplot(fp,fn,'fp','fn')
	#plt.axis([-50, 0, 0, 1])
	plt.grid(True)
	plt.title("Fp and Fn vs time to maneuver at thres="+str(manual_thres))
	plt.legend(bbox_to_anchor=(0.85, 0.9), loc=2, borderaxespad=0.)
	if show: plt.show()

	fig5 = plt.figure(5)
	getplot(precisionP,precisionN,'precisionP','precisionN')
	plt.grid(True)
	plt.axis([-50, 0, 0, 1])
	plt.title("Precision of positive and negative samples vs time to maneuver at thres="+str(manual_thres))
	plt.legend(bbox_to_anchor=(0.72, 0.7), loc=2, borderaxespad=0.)
	if show: plt.show()

	fig6 = plt.figure(6)
	getplot(recallP,recallN,'recallP','recallN')
	plt.grid(True)
	plt.axis([-50, 0, 0, 1])
	plt.title("Recall of positive and negative samples vs time to maneuver at thres="+str(manual_thres))
	plt.legend(bbox_to_anchor=(0.72, 0.7), loc=2, borderaxespad=0.)
	if show: plt.show()

	
	fig7 = plt.figure(7)
	plt.plot(thresholds,precisionP_thres,color='b',marker='x',label='precisionP')
	plt.plot(thresholds,precisionN_thres,color='r',marker='x',label='precisionN')
	plt.plot(thresholds,recallP_thres,color='b',marker='o',label='recallP')
	plt.plot(thresholds,recallN_thres,color='r',marker='o',label='recallN')
	plt.grid(True)
	plt.ylim([0, 1])
	plt.title("Precision and Recall of positive and negative samples vs thresholds at "+str(1.0*frame_index/10)+"secs before maneuver")
	plt.ylabel('Value')
	plt.xlabel('Thresholds')
	plt.legend(bbox_to_anchor=(0.1, 0.3), loc=2, borderaxespad=0.)
	if show: plt.show()

	fig8 = plt.figure(8)
	plt.plot(recallP_thres,precisionP_thres,color='b',marker='x')
	plt.ylabel('PrecisionP')
	plt.xlabel('RecallP')
	plt.title("Precision positive vs Recall positive across various thresholds")
	plt.plot(recallP_thres[0],precisionP_thres[0],marker='p',color='r',markersize=6)
	plt.annotate('thres=min' , xy = (recallP_thres[0],precisionP_thres[0]), textcoords = 'data')
	plt.plot(recallP_thres[-1],precisionP_thres[-1],marker='p',color='r',markersize=6)
	plt.annotate('thres=max' , xy = (recallP_thres[-1],precisionP_thres[-1]), textcoords = 'data')
	plt.ylim([0, 1])
	plt.grid(True)
	if show: plt.show()
	"""
	fig10 = plt.figure(10)
	ax = Axes3D(fig10)
	#ys = range(-eventWindow,0)
	#xs = thresholds
	xs,ys = np.meshgrid(thresholds,range(-eventWindow,0))
	print xs.shape
	print ys.shape
	#ax = plt.axes(projection='3d')
	ax.plot(xs, ys,f1_grid.T,'-b')
	plt.grid(True)
	if show: plt.show()
	"""



	"""Generating trip files with individual predicted labels"""
	#This is only possible if there are braking zones in your test file	
	if savePredLabelsTripWise and brakingZonesInY:

		#If the pred labels do not contain braking zones, we have to inject it into them
		if save_y.shape != save_pred_y.shape:
			tmp = np.copy(save_y)
			count =0
			for i in range(len(tmp)):
				if tmp[i,0]!=2: #replacing with predicted label when outside braking zone
					tmp[i] = save_pred_y[count]
					count+=1
			save_pred_y = tmp

		currDir = "./"
		tripSaveFolder = 'trip_wise_pred_label'
		if not os.path.isdir(currDir+tripSaveFolder):
			os.makedirs(currDir+tripSaveFolder)
		sessionIndexFile = currDir + "../sessionIndex.csv"
		sessionIndex = pd.read_csv(sessionIndexFile)
		tripsForTestDemo = sessionIndex.timestamp[sessionIndex.train_or_test_for_demo==1].tolist()
		tripsForTestDemo = [str(i).strip() for i in tripsForTestDemo]
		tripsForTestDemo.sort()
		count=0; fig=10

		save_pred_y[save_y==2] = 2 #setting prediction to 2 whenever braking is spotted
		
		for trip in tripsForTestDemo:
			if not os.path.isdir(currDir+'../../tripCsvFiles'):
				print "Sorry trip csv files not available for storing predicted labels trip wise"
				return

			data = pd.read_csv(currDir+'../../tripCsvFiles/'+trip+'_features_brake_agg.csv',usecols=["timestamp","brakeEvents","sparseBrakeEvents","emptyBrakeEvents","pbrk"])
			timedata = pd.DataFrame(data['timestamp'],columns=["timestamp"])
			brakePressure = data['pbrk'].values.tolist()
			pred_label_tosave = save_pred_y[count:count+int(len(timedata)/eventWindow)].reshape(-1)
			pred_label_tosave = np.append(pred_label_tosave,np.zeros(len(timedata)-len(pred_label_tosave)).astype(int)) #padding zeros for last imcomplete sample
			y_label_tocompare = save_y[count:count+int(len(timedata)/eventWindow)].reshape(-1)
			y_label_tocompare = np.append(y_label_tocompare,np.zeros(len(timedata)-len(y_label_tocompare)).astype(int)) #padding zeros for last imcomplete sample

			count = count + int(len(timedata)/eventWindow)

			
			pred_label_tosave[np.logical_and(pred_label_tosave==2, np.array(brakePressure)==0)] = 0  #removing anamolies
			y_label_tocompare[np.logical_and(y_label_tocompare==2, np.array(brakePressure)==0)] = 0  #removing anamolies
			#errorPoints = [i for i in range(len(pred_label_tosave)) if pred_label_tosave[i] != y_label_tocompare[i]]


			#slicing 5 sec brake events
			brakeEvents = data['sparseBrakeEvents'].values.tolist()
			#slicing 5 sec non brake events
			emptyBrakeEvents = data['emptyBrakeEvents'].tolist()


			sparseZeroCros = np.array([i for i, x in enumerate(brakeEvents) if x == 1])
			emptypoints = np.array([i for i, x in enumerate(emptyBrakeEvents) if x == 1])

			brakePressure = np.array(brakePressure)

			plt.figure(fig)        
			plt.plot(brakePressure, 'g-')
			plt.plot(sparseZeroCros,brakePressure[sparseZeroCros], 'r*', markersize=10, label='Brake events')
			plt.plot(emptypoints, brakePressure[emptypoints], 'g*', markersize=10, label='Non Brake events')
   			print "length of emptypoints :", len(emptyBrakeEvents)
   			tmp = 1.0*pred_label_tosave/10
   			a = np.array([i for i, x in enumerate(tmp) if x == 0.2])
   			plt.plot(a, tmp[a],'k+',markersize=3, label='Observed Brake Press')
   			a = np.array([i for i, x in enumerate(tmp) if x == 0.1])
   			plt.plot(a, tmp[a],'b+',markersize=3, label= 'Predicted Brake events')
   			a = np.array([i for i, x in enumerate(tmp) if x == 0])
   			plt.plot(a, tmp[a],'y+',markersize=3, label = 'Actual Brake events')
			#plt.plot(1.0*pred_label_tosave/10,'k.',markersize=2)
			plt.legend(loc='upper right')
			#plt.plot(errorPoints,1.0*y_label_tocompare[errorPoints]/10, 'y.',markersize=5)
			plt.title('Predictions and Errors for '+trip)
			plt.xlabel('Time (ms)')
			plt.ylabel('Brake Pressure')

			if show: plt.show()
			fig +=1
			
			
			timedata.insert(1,'pred_label',pred_label_tosave)
			timedata.insert(2,'actual_label',y_label_tocompare)
			timedata.to_csv(currDir+tripSaveFolder+'/'+trip+'_pred_label.csv',index=False)
			print "Saved trip wise pred labels of trip %s in the folder %s" %(trip,tripSaveFolder)




	return tp,fp,tn,fn,precisionP,recallP,precisionN,recallN












	"""
	fig5= plt.figure(5)
	plt.plot(fpr,tpr, color=line_color ,marker = 'x',label='tpr vs fpr')
	plt.title('ROC Curve')
	x1,x2,y1,y2 = plt.axis()
	plt.axis([x1,x2, 0, 1])
	if show: plt.show()

	
	plt.plot(range(-len(p_acc_svm),0),p_acc_svm,'r',label='svm')
	plt.plot(-len(p_acc_svm),p_acc_svm[0],marker='p',color='y',markersize=10)
	plt.plot(-1,p_acc_svm[-1],marker='p',color='y',markersize=10)
	plt.annotate('(%.3f)' % p_acc_svm[0], xy = (-len(p_acc_svm),p_acc_svm[0]), textcoords = 'data')
	plt.annotate('(%.3f)' % p_acc_svm[-1], xy = (-1,p_acc_svm[-1]), textcoords = 'data')

	plt.plot(range(-len(p_acc_svm),0),[0.6]*len(p_acc_svm),'k--',label='baseline')
	plt.legend(bbox_to_anchor=(0.75, 0.7), loc=2, borderaxespad=0.)
	plt.ylim(0.5,1)   #to show over a range of 50 frames
	"""

	"""
	groundtruth_y = np.zeros(y.shape[0])
	predict_y = np.zeros(y.shape[0])
	for i in range(y.shape[0]):
		c1 = Counter(pred_y_rm[i,:])
		value1, _= c1.most_common()[0]
		c2 = Counter(y[i,:])
		value2, _= c2.most_common()[0]
		predict_y[i] = value1
		groundtruth_y[i] = value2
	hit_final = 0.0
	for j in range(len(groundtruth_y)):
		if (int(predict_y[j])==int(groundtruth_y[j])):
			hit_final+=1
	print "framewise accuracy: ",hit_final/len(groundtruth_y)
	
	return
	"""
	
if __name__=='__main__':
	
	processResults('p_labels.txt','gt_labels.txt','raw_p_labels.txt','b',0)
	
	plt.show()





