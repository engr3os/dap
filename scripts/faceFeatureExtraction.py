import os
from scipy import misc
import numpy as np

print "Started"
#sourcepath = "/mnt/disk3/fy15_sakiyomiagent_datacollection/polysync/polysync_processed"
sourcepath = "/mnt/disk1/polysync/polysync_processed"
savepath_main_folder = "../processed/"
savepath_folder = "../processed/face_points/"
dirlist = os.listdir(sourcepath)
print dirlist

#creating necessary folders
if not os.path.isdir(savepath_main_folder[:-1]):
	os.makedirs(savepath_main_folder[:-1])
if not os.path.isdir(savepath_folder[:-1]):
	os.makedirs(savepath_folder[:-1])

duoface = ["1458254694982667"]  #to accomodate for some trips where the folder is "head_cam instead of face_cam"

for dir in dirlist:
	if dir in duoface:
		faceCamFolder = "/head_cam/"
	else:
		faceCamFolder = "/face_cam/"
	if os.path.isdir(sourcepath+"/"+dir):
		if os.path.isdir(sourcepath + "/" + dir + faceCamFolder[:-1]):
			imgpath = sourcepath + "/" + dir + faceCamFolder
			imgsavepath = savepath_folder + dir + "_processed/"

			
			if not os.path.isdir(imgsavepath[:-1]):
				os.makedirs(imgsavepath[:-1])

			args = "ls " + imgpath + " | grep  .jpg$ | wc -l"  #argument for rotating images
			print args
			count = os.system(args)
			print count
			#i = 0
			print "processing %s" % dir
			for pic in os.listdir(sourcepath + "/" + dir + faceCamFolder):			
				if pic.endswith(".jpg"):
					try:
						img = misc.imread(imgpath+pic)
					except Exception: 
						continue
					flipim = np.flipud(img)
					flipim = np.fliplr(flipim)
					misc.imsave(imgsavepath+pic, flipim)  #saving rotated images temporarily
					

			if not os.path.isdir(savepath_folder + dir + "Feature"):
				os.makedirs(savepath_folder + dir + "Feature")
			args = "../CLM-framework-master/bin/SimpleCLMImg -fdir " + imgsavepath + " -ofdir "+ savepath_folder + dir + "Feature/ -clmwild"
			os.system(args)
			print "Finished processing %s" %dir
			os.system("rm -rf " + imgsavepath[:-1])  #deleting rotated images

print 'Done Flipping and Extracting features'