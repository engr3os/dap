# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:41:06 2016

@author: vijay
"""

import os
from data_labeling_tool import data_extractor

path = "/mnt/disk1/polysync/polysync_processed"
savepath = '../pickleFiles/'

if not os.path.isdir(savepath[:-1]):
	os.makedirs(savepath[:-1])


dirs = os.listdir(path)

for dir in dirs:
	if os.path.isdir(path + "/"+dir) and os.path.isfile(path+"/"+dir+"/CAN_log.json"):
		#os.system("./data_labeling_tool/data_extractor.py "+ path + "/"+dir)
		data_out, folder = data_extractor.extractor(path)
		pickle.dump(data_out, open(savepath+folder+'_extr.pkl','wb'))
		print "Generated pickle for folder: ", folder

