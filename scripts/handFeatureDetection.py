import os
from scipy import misc
from subprocess import call
import numpy as np
import argparse
from operator import itemgetter

def parse_args():
	"""Set the GPU parameter"""
	parser = argparse.ArgumentParser(description='Automation for FRCNN demo')	
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
	args = parser.parse_args()
	return args




print "Started"
#sourcepath = "/mnt/disk3/fy15_sakiyomiagent_datacollection/polysync/polysync_processed"
sourcepath = "/mnt/disk1/polysync/polysync_processed"
processingfilepath = "/home/tzhou/Software/py-faster-rcnn/tools/demo_modified.py"
dirlist = os.listdir(sourcepath)
#print dirlist

#done = ["1461967704454056", "1462554563684426", "1461621235238498", "1460673068694649", "1453410573148146", "1463077101808539"]
folderlist = []

for dir in dirlist:
	if os.path.isdir(sourcepath + "/"+dir):
		if os.path.isdir(sourcepath + "/" + dir + "/hand_cam"):
			folderlist.append([dir,"hand_cam"])
		elif os.path.isdir(sourcepath + "/" + dir + "/handcam"):
			folderlist.append([dir,"handcam"])

folderlist = sorted(folderlist, key=itemgetter(0))
print folderlist
#folder = folderlist[0]
args = parse_args()
gpu_id = args.gpu_id

n = int(len(folderlist)/4)

if gpu_id == 0:
	folderlistnew = folderlist[0:n]
elif gpu_id == 1:
	folderlistnew = folderlist[n:2*n]
elif gpu_id == 2:
	folderlistnew = folderlist[2*n:3*n]
elif gpu_id == 3:
	folderlistnew = folderlist[3*n:]

for folder in folderlistnew:
#print "Processing folder %s" %folder[0]
	args = "python " + processingfilepath + " --gpu " + str(gpu_id) + " --folderName " + folder[0] + " --handFolder " + folder[1]
	try:
		os.system(args)
	except Exception:
		break

print "Done"
