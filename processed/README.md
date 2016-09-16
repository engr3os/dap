# Extract CAN data, face features and hand features from raw face and hand images

**Note:** Due to the large size of the data, I only included histograms and extracted hand feature text files in this repo. For full data, go to *10.0.0.7/home/vijay/processed/*.

Also you need to be logged into 10.0.0.7 (artemis). This utilizes the dataset available at *10.0.0.7/mnt/disk1/polysync/polysync_processed/*

#### Pickle CAN, obd, object logs from 10.0.0.7/mnt/disk1/polysync/polysync_processed:
* cd ../scripts
* python [generatePickle.py](../scripts/generatePickle.py)
    * This will use Tobi's CAN data extractor tool to generate pickle files for each trip from files from *10.0.0.7/mnt/disk1/polysync/polysync_processed/*
	* The pickle files will be stored in [../pickleFiles](../pickleFiles)
	* You might need sudo access for saving the pickle files


#### Face Points extraction:
* cd ../scripts
* python [faceFeatureExtraction.py](../scripts/faceFeatureExtraction.py)
	* Rotates the images in *10.0.0.7/mnt/disk1/polysync/polysync_processed* folder by folder and runs ``` 
"../CLM-framework-master/bin/SimpleCLMImg -fdir " + rotated_dir + " -ofdir ../processed/face_points" + dir + "Feature/ -clmwild" ```
to output the face feature points from all the images to the folder [../processed/face_points](face_points)
	* if you want to save the post-processed images too, add -iodir parameter

#### Histogram calculation from face features:
* cd ../scripts
* python [histogramCreation.py](../scripts/histogramCreation.py)
    * This creates the angular and 2-D histograms of consequtive point files (or timestamps) for each of the folders in [/face_points](face_points)
    * Creates a csv file for each trip inside the folder [/histograms](/histograms)

#### Hand Feature extraction:
* cd ../scripts
* python [handFeatureDetection.py](../scripts/handFeatureDetection.py) --gpu (0 or 1 or 2 or 3)
	* This will access the file at *10.0.0.7/home/tzhou/Software/py-faster-rcnn/tools/demo_modified.py* which runs a fast RNN using a pre trained model to identify the hand locations
	* Running this file will only give the hand locations but will not tell anything about number of hands on wheel etc.
	* The hand locations are saved in the folder */home/vijay/processed/handfeatures*. Modify the location in file if required. If you do so you will need to modify in the below file too

* **Next step is to convert this hand locations to parameters like 'numHandsOnWheel', 'leftMoving' etc.**
* Login to tzhou@10.0.0.165 -X using pwd toyota1234
* cd /home/tzhou/Workspace/interns/BrakePrediction/src
* ```python HandCamReader_modified.py --folder "sessionNumber" --copydata (use 1 if you need to copy from 10.0.0.7/mnt/disk1/polysync/polysync_processed else 0) --res (use 1 if you want to remove existing data and recopy from that polysync_processed location else 0)```
	* The process takes you through a set of steps to choose the points on the steering wheel for reference when using the extracted hand location
	* The computed data is stored in */home/tzhou/Workspace/interns/BrakePrediction/data/*
* I manually copied this data to */home/vijay/processed/histograms_processed/* for future use

