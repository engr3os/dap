import sys
import numpy as np
import tensorflow as tf
#from tensorflow.models.rnn import rnn, bidirectional_rnn, rnn_cell
from tensorflow.python.ops.constant_op import constant
import cPickle
import math
from scipy.sparse import diags
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pdb
from tensorflow.python.ops import math_ops
import postprocessing


if __name__ == '__main__':
	
	# Date Loading
	"""
	X_tr = np.load('/home/rguo/data/rawFeature/X_train20zero.npy')
	Y_tr = np.load('/home/rguo/data/rawFeature/Y_train20zero.npy')
	X_te = np.load('/home/rguo/data/rawFeature/X_test20zero.npy')
	Y_te = np.load('/home/rguo/data/rawFeature/Y_test20zero.npy')
	"""
	
	X_tr = np.load('../../data_for_network_balanced_demo/X_train_20pca_balanced_norm.npy')
	Y_tr = np.load('../../data_for_network_balanced_demo/Y_train_balanced.npy')
	X_te = np.load('../../data_for_network_balanced_demo/X_test_20pca_balanced_norm.npy')
	Y_te = np.load('../../data_for_network_balanced_demo/Y_test_balanced.npy')
	
	"""
	X_tr = np.load('/home/vijay/data_for_network_balanced/X_train_20pca_balanced_norm.npy')
	Y_tr = np.load('/home/vijay/data_for_network_balanced/Y_train_balanced.npy')
	X_te = np.load('/home/vijay/data_for_network_balanced/X_test_20pca_balanced_norm.npy')
	Y_te = np.load('/home/vijay/data_for_network_balanced/Y_test_balanced.npy')
	"""
	"""
	X_tr = np.load('/home/vijay/data_for_network_balanced/X_train_balanced_norm.npy')
	Y_tr = np.load('/home/vijay/data_for_network_balanced/Y_train_balanced.npy')
	X_te = np.load('/home/vijay/data_for_network_balanced/X_test_balanced_norm.npy')
	Y_te = np.load('/home/vijay/data_for_network_balanced/Y_test_balanced.npy')
	"""

	"""set this parameter to include or not include braking zones (zones where the brake pressure is non-zero"""
	include_braking_zones = 1

	
	len_samples = X_tr.shape[2]
	n_steps = X_tr.shape[1]

	"""This is to set whether to load a previously trained model or not"""
	load_model = 0
	if load_model:
		print "You have set to use the presaved model"

	save_model_path = "./trainedModel/"
	model_name = "BiRNN_GRU_model.ckpt"

	"""When you have zones in data with non zero brake pressure (braking zones), Y_te will contain Nones. This snippet will replace all the Nones to 0 for the network to understand"""
	saveY_te = np.copy(Y_te)
	if 2 in Y_te:  #meaning braking zone found	
		if include_braking_zones:
			print "INCLUDING BRAKING ZONES IN TESTING..."			
			Y_te[Y_te==2] = 0  #setting y_test to zero in the braking points (ignore live test accuracy and wait for metrics if you do so)
		elif not include_braking_zones:
			print "NOT INCLUDING BRAKING ZONES IN TESTING..."
			tmp = (Y_te==2).flatten()
			X_te = X_te.reshape([-1,len_samples])[np.logical_not(tmp)].reshape([-1,n_steps,len_samples])
			Y_te = Y_te.flatten()[np.logical_not(tmp)].reshape([-1,n_steps])

	print "shape of X_tr: ",X_tr.shape
	print "shape of Y_tr: ",Y_tr.shape
	print "shape of X_te: ",X_te.shape
	print "shape of Y_te: ",Y_te.shape
	print "Y_tr: ",Y_tr[1,:]
	print "Y_tr: ",Y_tr[10,:]
	print "Y_tr: ",Y_tr[100,:]
	print "Y_tr: ",Y_tr[150,:]
	print "Y_tr.max: ",np.max(Y_tr)
	print "Y_tr.min: ",np.min(Y_tr)
	num_train = X_tr.shape[0]
	num_test = X_te.shape[0]


	num_classes = 2

	# Network Parameter Configuration
	learning_rate = 0.0002
	learning_decay = 1
	decay_delay_step = 550*num_train
	training_iters = 900*num_train
	num_batch = 1
	train_batch_size = num_train/num_batch
	display_step = 30
	display_pr_step = 120
	max_grad_norm = 10
	lossratio = 5.0

	#Network Settings
	num_layers = 1 # RNN layers. The Bidirectional RNN layer already exist
	n_input = X_tr.shape[-1]
	n_hidden = 128
	n_classes = num_classes
	thresholding = 0.0 # if the estimation confidence of maneuer is lower than the threshold, then assume it going straight
	target_names = ['Non-Brake','Brake']
	#-----------------------------------------------------------------------------------------------------------------------------#
	#One-Hot-Code Encoder
	def Dense_To_One_Hot_Encoder(_labels_dense, _num_classes):
		"Convert class labels from scalars to one-hot vectors. Input: [num_labels,1]; Output: [num_labels, num_classes]"
		_num_labels = _labels_dense.shape[0]
		_index_offset = np.arange(_num_labels) * _num_classes
		_labels_one_hot = np.zeros((_num_labels, _num_classes))
		ind = _index_offset + _labels_dense.ravel()
		#pdb.set_trace()
		_labels_one_hot.flat[ind.astype(int)] = 1
		return _labels_one_hot
	#-----------------------------------------------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------------------------------------------#
	# Next Batch Generator
	def Next_Batch(_X, _Y, _batch_size, _step): # _X.shape: [num_samples, n_steps, n_input]; _Y.shape: [num_samples, n_steps]
		assert _X.shape[0] == _Y.shape[0],"Number of samples and number of labels should be the same!"
		assert _batch_size <= _X.shape[0],"Batch size already larger than the whole data size. Reduce the batch size!"
		if _step*_batch_size%_X.shape[0]==0:
			batch_xs_original = _X[(_step-1)*(_batch_size)%(_X.shape[0]):_X.shape[0],:,:]
			batch_ys_original = _Y[(_step-1)*(_batch_size)%(_X.shape[0]):_X.shape[0],:]
		elif (_step-1)*_batch_size%_X.shape[0]+_batch_size > _X.shape[0]:			
			batch_xs_original = _X[_X.shape[0]-_batch_size:_X.shape[0],:,:]
			batch_ys_original = _Y[_X.shape[0]-_batch_size:_X.shape[0],:]
		else:
			batch_xs_original = _X[(step-1)*(_batch_size)%(_X.shape[0]):step*_batch_size%_X.shape[0],:,:]
			batch_ys_original = _Y[(step-1)*(_batch_size)%(_X.shape[0]):step*_batch_size%_X.shape[0],:]
		return batch_xs_original, batch_ys_original
			
	#-----------------------------------------------------------------------------------------------------------------------------#		

	#-----------------------------------------------------------------------------------------------------------------------------#	
	#Precision and Recall Calculation
	def Precision_and_Recall_and_Accuracy(_predict_labels,_ground_truth_labels):	
		"""Action Type: || end_action || lchange || rchange || lturn || rturn || Straight ||"""
		"""Labels:      ||      0     ||     1   ||     2   ||   3   ||   4   ||     5    ||"""
		"""label 0 is the dummy label. only actions 1,2,3,4 are maneuvers, action 5 is driving straight"""
		"""(i) true prediction: tp. correct maneuver prediction"""
		"""(ii) false prediction: fp. predict a maneuver but driver did another maneuver"""
		"""(iii) false postive prediction: fpp. predict maneuver but go straignt"""
		"""(iv) missed prediction: mp. predict going straight but driver maneuver"""
		"""Definition: Precision = tp/(tp+fp+fpp);   Recall = tp/(tp+fp+mp)"""
		#if tf.size(_predict_labels).eval()!=tf.size(_ground_truth_labels).eval():
		assert _predict_labels.shape[1]==_ground_truth_labels.shape[1], "Lengths of predict labels and groundtruth labels must be the same !"
		_length = _predict_labels.shape[1]
		_Accuracy = hit_count/(total_num+0.00001)
		_num_classes = _predict_labels.max()-__predict_labels.min()+1
		conf_mat = Confusion_Matrix_Computation(_predict_labels, _ground_truth_labels, _num_classes)
		tp = np.asarray(np.diag(conf_mat).flatten(),dtype='float')
		pred_class = np.asarray(np.sum(conf_mat, axis=0).flatten(),dtype='float');
		act_class = np.asarray(np.sum(conf_mat, axis=1).flatten(),dtype='float')
		_Precision = np.nanmean(tp[1:-1]/pred_class[1:-1])
		_Recall = np.nanmean(tp[1:-1]/act_class[1:-1])
		_Accuracy = tp.sum()/conf_mat.sum()
		return _Precision, _Recall, _Accuracy
	#-----------------------------------------------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------------------------------------------#
	def PRCmetric(_predict_labels,_ground_truth_labels):
		assert _predict_labels.shape[1]==_ground_truth_labels.shape[1],"Lengths of predict labels and groundtruth labels must be the same !"
		
	#-----------------------------------------------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------------------------------------------#
	# Confusion Matrix
	def Confusion_Matrix_Computation(_predict_labels, _ground_truth_labels, _num_classes):
		assert _predict_labels.shape[1]==_ground_truth_labels.shape[1],"Lengths of predict labels and groundtruth labels must be the same !"
		_length = _predict_labels.shape[1]	
		confusion_mat = np.zeros((_num_classes,_num_classes), dtype = np.float)
		for i in range(0, _length):
			confusion_mat[int(_predict_labels[0,i]),int(_ground_truth_labels[0,i])]+=1
		return confusion_mat
	#-----------------------------------------------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------------------------------------------#
	# Add the threshold on the softmax output for accurate prediction
	def thresholdLabeling(_classification, _pred_labels, _thresholding):
		assert _classification.shape[0] == _pred_labels.shape[1],"Number of samples in sequence and number of labels should be the same!"
		_n_steps = _classification.shape[0]
		for i in range(0,_n_steps):
			temp = [0]*_classification.shape[1]
			temp = _classification[i,:]
			temp = softmax(temp)
			_pred_labels[0,i] = 5 if (temp[_pred_labels[0,i]]<_thresholding) else (_pred_labels[0,i])
		return _pred_labels
	#-----------------------------------------------------------------------------------------------------------------------------#

	#-----------------------------------------------------------------------------------------------------------------------------#
	#Plot Confusion Matrix
	def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
    		plt.title(title)
    		plt.colorbar()
    		tick_marks = np.arange(len(target_names))
    		plt.xticks(tick_marks, target_names, rotation=30)
    		plt.yticks(tick_marks, target_names)
    		plt.tight_layout()
    		plt.ylabel('True label')
    		plt.xlabel('Predicted label')
	#-----------------------------------------------------------------------------------------------------------------------------#
				
	#---------------------------------------------Network Configuration-----------------------------------------------------------#
	#TensorFlow Graph Structure
	x = tf.placeholder("float",[None, n_steps, n_input])
	y = tf.placeholder("float",[None, n_steps, n_classes])
	batch_size = tf.placeholder(dtype=tf.int32)
	class MultiRecurrentNN(object):
		def __init__(self,_num_layers, _n_hidden, _n_output):
			# Define lstm cells with tensorflow
    		# Forward direction cell
			self.lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(_n_hidden, forget_bias=1.0) #lstm_fw_cell = rnn_cell.GRUCell(n_hidden) #
    		# Backward direction cell
			self.lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(_n_hidden, forget_bias=1.0) #lstm_bw_cell = rnn_cell.GRUCell(n_hidden) #
			# Top layer cell
			lstm_cell = tf.nn.rnn_cell.GRUCell(_n_hidden) #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
			self.multi_RNN = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * _num_layers)
			#Define Weights and Connections
			self.weights = {
				'out':tf.Variable(tf.random_normal([_n_hidden, _n_output])),
				'loss':tf.Variable(tf.random_normal([_n_hidden, 1]))
			}
			self.biases = {
				'out':tf.Variable(tf.random_normal([_n_output])),
				'loss':tf.Variable(tf.random_normal([1]))
			}
		def compute(self, _X, _n_input, _n_steps, _istate_fw, _istate_bw, _istate_rnn):
			# input shape: (batch_size, n_steps, n_input)
			_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
			# Reshape to prepare input to hidden activation
			_X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
			_X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
			# Get Bidirectional lstm cell output
			#"""
			BiRNN_outputs, _istate_fw, _istate_bw = tf.nn.bidirectional_rnn(self.lstm_fw_cell, self.lstm_bw_cell, _X,
		                                        	initial_state_fw=_istate_fw,
		                                        	initial_state_bw=_istate_bw)
			"""
			BiRNN_outputs, _istate_fw = tf.nn.rnn(self.lstm_fw_cell,_X,
													initial_state=_istate_fw)
			"""
			outputs_rnn, _istate_rnn = tf.nn.rnn(self.multi_RNN, BiRNN_outputs,initial_state=_istate_rnn, dtype=tf.float32)
			outputs_rnn = tf.reshape(tf.concat(0, outputs_rnn), [-1, n_hidden])
			logit = tf.nn.xw_plus_b(outputs_rnn, self.weights['out'], self.biases['out'])
			loss_weight = tf.sigmoid(tf.nn.xw_plus_b(outputs_rnn, self.weights['loss'], self.biases['loss']))
			return logit, loss_weight, _istate_fw, _istate_bw, _istate_rnn

	#pred_y = BiRecurrentNN(x, istate_fw, istate_bw, weights, biases, batch_size, n_steps)
	
	# Exponential loss
	#strength = np.exp(-(np.arange(n_steps,0,-1)-1)/lossratio)
	#print "length of strength: ",strength
	#loss_weight_array = np.repeat(strength,	train_batch_size)
	#loss_weight = tf.constant(loss_weight_array,dtype=tf.float32)
	multiRecurrentNN = 	MultiRecurrentNN(num_layers, n_hidden, n_classes)
	istate_fw = multiRecurrentNN.lstm_fw_cell.zero_state(batch_size,tf.float32)
	istate_bw = multiRecurrentNN.lstm_bw_cell.zero_state(batch_size,tf.float32)
	istate_rnn = multiRecurrentNN.multi_RNN.zero_state(batch_size,tf.float32)
	pred_y, loss_weight, istate_fw, istate_bw, istate_rnn = multiRecurrentNN.compute(x, n_input, n_steps, istate_fw, istate_bw, istate_rnn)

	# Define optimizer with norm limitation
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_y, tf.reshape(tf.transpose(y,[1,0,2]), [-1, num_classes]))) # Softmax loss
	cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred_y, tf.reshape(tf.transpose(y,[1,0,2]), [-1, num_classes]), loss_weight))/tf.reduce_mean(loss_weight) # Softmax loss
	#loss_mul = []
	#spred_y = tf.split(0, n_steps, pred_y)
	#sy = tf.split(0, n_steps, tf.reshape(y, [-1, num_classes]))
	#sloss_weight = tf.split(0, n_steps, loss_weight)
	#for i,j,k in zip(spred_y, sy, sloss_weight):
	#	loss_mul.append(tf.mul(k,tf.nn.softmax_cross_entropy_with_logits(i,j)))
	#loss_mul = tf.concat(0, loss_mul)
	#cost = tf.reduce_mean(loss_mul)/tf.reduce_mean(tf.square(loss_weight)) # Softmax loss
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # Adam Optimizer
	train_op = optimizer.apply_gradients(zip(grads, tvars))

	#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred_y,1), tf.argmax(tf.reshape(tf.transpose(y,[1,0,2]), [-1, num_classes]),1)) ## Pay attention!!!!!!!
	#correct_pred = tf.equal(tf.argmax(tf.reshape(pred_y,[num_classes, -1]),0), tf.argmax(tf.reshape(y,[num_classes, -1]),0)) ## Pay attention!!!!!!!
	#correct_pred = tf.equal(tf.argmax(pred_y,1), tf.argmax(y,1)) ## Pay attention!!!!!!!
	#print "correct_pred_shape: ",tf.reshape(correct_pred,[1,-1]).eval()
	
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	# Create a summary to monitor cost function
	tf.scalar_summary("loss", cost)

	# Merge all summaries to a single operator
	merged_summary_op = tf.merge_all_summaries()
	#-----------------------------------------------------------------------------------------------------------------------------#

	#-------------------------------------------------------------- Testing-------------------------------------------------------#
	#Function for testing
	def get_predictions(sess,X_test,Y_test,test_batch_size,n_steps,num_classes,test_acc_final,p_labels_final,raw_p_labels_final):
		test_count = 0
		p_labels = np.empty((0,n_steps))
		raw_p_labels = np.empty((0,n_steps,num_classes))
		test_Acc = 0.0
		test_Acc_array = []
		#print "-------------------------------testing phase starts----------------------------"
		while test_count*test_batch_size < n_test :
			batch_xt, batch_yt = Next_Batch(X_test, Y_test, test_batch_size, test_count)
			#print "batch_xt.shape: ",batch_xt.shape
			#print "batch_yt.shape: ",batch_yt.shape
			x_test_sample = batch_xt #.reshape((n_steps, test_batch_size, n_input))
			y_test_sample = np.empty((0,n_steps*num_classes))
			for k in range(0, batch_yt.shape[0]):
				temp = Dense_To_One_Hot_Encoder(batch_yt[k,:],num_classes)
				y_test_sample = np.vstack((y_test_sample,temp.reshape(1,batch_yt.shape[1]*num_classes))) 
			y_test_sample = y_test_sample.reshape((-1,batch_yt.shape[1],num_classes))

			raw_pred_labels = sess.run(tf.sigmoid(pred_y),feed_dict={x: x_test_sample, y: y_test_sample,
															batch_size:test_batch_size})

			raw_pred_labels = raw_pred_labels.reshape([n_steps,-1,num_classes]).transpose([1,0,2]).reshape([-1,n_steps,num_classes])
			raw_p_labels = np.vstack((raw_p_labels,raw_pred_labels))


			pred_labels = sess.run(tf.argmax(pred_y,1), feed_dict={x: x_test_sample, y: y_test_sample,  
															batch_size:test_batch_size})
			#pdb.set_trace()

			pred_labels = pred_labels.reshape([n_steps,-1]).transpose([1,0]).reshape([-1,n_steps])
			p_labels = np.vstack((p_labels,pred_labels))
			test_count+=1
			test_Accuracy = sess.run(accuracy, feed_dict={x: x_test_sample, y: y_test_sample,  
											 	      batch_size:test_batch_size})
			test_Acc = test_Acc + test_Accuracy
			test_Acc_array.append(test_Accuracy)
		
		#print "testing accuracy: ",test_Acc/len(test_Acc_array)
		if test_Acc/len(test_Acc_array) > test_acc_final:
			test_acc_final = test_Acc/len(test_Acc_array)
			p_labels_final = p_labels
			raw_p_labels_final = raw_p_labels
		return p_labels_final, raw_p_labels_final, test_acc_final, test_Acc, test_Acc_array

	#------------------------------------------------------------Training---------------------------------------------------------#

	# Initializing the variables
	init = tf.initialize_all_variables()
	# Create the saver
	saver = tf.train.Saver()
	# Preparing the testing data and labels
	X_test = X_te
	Y_test = Y_te
	test_batch_size = Y_test.shape[0]
	n_test = num_test
	p_labels_final = np.empty((0,n_steps))
	raw_p_labels_final = np.empty((0,n_steps,num_classes))
	test_acc_final = 0.0
	print "X_test.shape: ",X_test.shape
	print "Y_test.shape: ",Y_test.shape
	# Launch the graph
	if not load_model:
		with tf.Session() as sess:
			"""training batch equals the whole size of trainset"""
			train_batch_size = num_train/num_batch
			sess.run(init)
			summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph=sess.graph)
			step = 1
			acc = 0.0
			# Keep training until reach max iterations
			while step*train_batch_size<training_iters and acc <0.93: #step * batch_size < training_iters:
				if step>=decay_delay_step:
					learning_rate = learning_rate*learning_decay
				batch_xs, batch_ys = Next_Batch(X_tr, Y_tr, train_batch_size, step)
				# Reshape data to get 7 seq of 13 elements
				#batch_xs = batch_xs.reshape((n_steps, train_batch_size, n_input))

				# Convert batch_ys into One_Hot Code
				batch_ys_one_hot = np.empty((0,n_steps*num_classes))
				for j in range(0,batch_ys.shape[0]):
					temp = Dense_To_One_Hot_Encoder(batch_ys[j,:],num_classes)
					batch_ys_one_hot = np.vstack((batch_ys_one_hot,temp.reshape(1,batch_ys.shape[1]*num_classes)))
				#print 'batch_ys_one_hot_shape: ',batch_ys_one_hot.shape
				# Fit training using batch data
				batch_ys_one_hot = batch_ys_one_hot.reshape((-1,batch_ys.shape[1],num_classes)) 
				sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys_one_hot, batch_size: train_batch_size})	
				# Write logs at every iteration
				summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys_one_hot,   
											 batch_size: train_batch_size})

				summary_writer.add_summary(summary_str, step*train_batch_size)
				if step % display_step == 0:
					# Calculate batch accuracy
					acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys_one_hot,  
									batch_size: train_batch_size})
					# Calculate batch loss
					loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys_one_hot,  
									 batch_size: train_batch_size})
					#get predictions
					p_labels_final, raw_p_labels_final, test_acc_final, test_Acc, test_Acc_array = get_predictions(sess,X_test,Y_test,test_batch_size,n_steps,num_classes,
																														test_acc_final,p_labels_final,raw_p_labels_final)

					print "Iter " + str(step*train_batch_size) + "/" + str(training_iters)+ ", Minibatch Loss= " + "{:.6f}".format(loss) + \
						", Training Accuracy= " + "{:.5f}".format(acc)+ ", Testing Accuracy= " + "{:.5f}".format(test_Acc/len(test_Acc_array))

				step += 1
			# save the best trained network configuration
			save_path = saver.save(sess, save_model_path+model_name)
			print("Model saved in file: %s" % save_path)
			sess.close()
			print "Optimization Finished!"

	else: #using saved model
		with tf.Session() as sess:
			sess.run(init)
			saver.restore(sess, save_model_path+ model_name)
			print "Restored saved model "+ model_name
			p_labels_final, raw_p_labels_final, test_acc_final, test_Acc, test_Acc_array = get_predictions(sess,X_test,Y_test,test_batch_size,n_steps,num_classes,
																														test_acc_final,p_labels_final,raw_p_labels_final)
			sess.close()
			#Note: The output test accuracy achieved after using the saved model might be lower than the test accuracy achieved during actual training since 

	print "Predict Labels: ",p_labels_final
	np.savetxt("p_labels.txt",p_labels_final,fmt='%i')
	print "Ground Truth Labels: ",Y_test
	np.savetxt("gt_labels.txt",saveY_te,fmt='%i')    #saving the actual labels including 2's if present for postprocessing
	print "Raw Pred Labels[0,:num_classes]: ",raw_p_labels_final[0,:num_classes]
	np.savetxt("raw_p_labels.txt",raw_p_labels_final.reshape([-1,n_steps*num_classes]),fmt='%1.4f')
	np.save("raw_p_labels.npy",raw_p_labels_final)
	print "Final Testing Accuracy= " + "{:.5f}".format(test_acc_final)
	col_acc = (p_labels_final==Y_test).mean(axis=0)
	print "Per column accuracy: ", col_acc

	#pdb.set_trace()
	
	###plotting###

	plt.plot((np.arange(-n_steps,0)+1), col_acc, '-ro')
	plt.xlabel('Time to Manuever (x0.1sec)')
	plt.ylabel('Accuracy')
	plt.title('Preliminary Result (Ignore and compute metrics if braking zones are included)')
	#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
	plt.axis([-n_steps, 0, 0.5, 1])
	plt.show()
		#print "The thresholding value: "+ "{:.3f}".format(thresholding)
		#print "Testing Accuracy:" + "{:.6f}".format(acc) + ", Precision = "+ "{:.5f}".format(precision) + ", Recall = "+ "{:.5f}".format(recall)
	postprocessing.processResults('./p_labels.txt','./gt_labels.txt','./raw_p_labels.txt')
	plt.show()
		
