# Driver Intention Prediction

This code base is for the driver activity prediction project. Please contact Vijay (*achintal@ucsd.edu*) for any issues
* If you would want to generate train/test the network or regenerate data for the network, the instructions are below
* If you want to regenerate the csv files for the trips from the pickle files, read the [README.md](scripts/README.md) under [/scripts](scripts)
* If you would want to regenerate face and hand features from raw face and hand cam images read the [README.md](processed/README.md) under [/processed](processed)

#####
## Prerequisites 
* Have [python 2.3](https://www.python.org/download/releases/2.3/) or above installed (including sklearn, numpy and pandas packages)
* Have [OpenCV 3.1](http://opencv.org/opencv-3-1.html) installed if you want to extract face features from scratch
* Have [Tensorflow 0.8](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html) installed for deep learning 

#####
## Training and testing the network
* Data for balanced set is available in [/data_for_network_balanced](data_for_network_balanced) and data for demo is available in [/data_for_network_balanced_demo](/data_for_network_balanced_demo)

#### For Toyota data (both for balanced testing and demo):
* cd scripts/network_training/
* python [MultiRNN_brake_revised_tobi.py](scripts/network_training/MultiRNN_brake_revised_tobi.py)
	* Set the proper location of the trip generated numpy files
	* You can set whether to used a saved trained model or run freshly by setting the **'load_model'** parameter
	* Network is trained over the data and provides multiple plots using the [postprocessing.py](scripts/network_training/postprocessing.py) script
	* The trained model is stored in [./trainedModel](scripts/network_training/trainedModel). The script also stores the predicted, actual and raw probability values in .txt files
* __Note (Imp)__: If you are using test data including braking zones _(zones where brake pressure is non zero)_ then:
	* The test accuracy displayed while training isn't correct. Wait for the metrics to be displayed once training is done 
	* The metrics are calculated only for the non-braking zones _(zones where brake pressure is zero)_

#### For brain4cars data:
* cd scripts/network_training/
* python [MultiRNN_brake_revised_tobi_brain4cars.py](scripts/network_training/MultiRNN_brake_revised_tobi_brain4cars.py)
	* Set the proper location of the brain4cars generated numpy files
	* You can set whether to used a saved trained model or run freshly by setting the **'load_model'** parameter
	* Network is trained over the data and plots precision and recall curve for all classes over the 5 sec interval
	* Network is trained over the data and provides multiple plots using the [postprocessing.py](scripts/network_training/postprocessing.py) script
	* The trained model is stored in [./trainedModel](scripts/network_training/trainedModel). The script also stores the predicted, actual and raw probability values in .txt files


#####


## Regenerating data for the network

#### Generate data for the network from all trips: [Generate balanced normalized/non-normalized, PCA/non-PCA data for network]

* cd scripts
* python [dataCreation.py](scripts/dataCreation.py)
	* The features to include can be selected in the file. The number of PCA components can also be selected in the file
	* The script will take files *"xxx_feature_brake_agg.csv"* from the folder [../tripCsvFiles](tripCsvFiles)
	* Multiple kinds on numpy files are created. The file names are self explanatory. The numpy files are stored in [../data_for_network_balanced](data_for_network_balanced)

#### Generate network data for demo with whole length trips in test:
* cd scripts
* Set the training and testing trips in [*sessionIndex.csv*](scripts/sessionIndex.csv] under the column *"train_or_test_for_demo"*
* python [dataCreationForDemo.py](scripts/dataCreationForDemo.py):
	* **Imp:** Set whether you want to include braking zones *(zones where the brake pressure is non zero)* in the test set or not in the script
	* You can set the number of PCA components too in the file
	* The script will take files *"xxx_feature_brake_agg.csv"* from the folder [../tripCsvFiles](tripCsvFiles)
	* Multiple kinds on numpy files are created. The file names are self explanatory. The numpy files are stored in [../data_for_network_balanced_demo](data_for_network_balanced_demo)

#### Generate network data from brain4cars data:
* cd scripts
* The CSV files of brain4cars data prep is in the folder [../brain4cars_data](brain4cars)
* python [brain4cars_dataprep.py](scripts/brain4cars_dataprep.py)
	* The scripts takes data from ../brain4cars_data and generates shuffled non PCA and non normalized numpy files
	


