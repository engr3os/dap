## Regenerate CSV files for each trip

#### Create Trip Csv files from pickle files:
* **Note**: Due to the huge file size, the pickle files are stored in *10.0.0.7/home/vijay/pickleFiles*. Makesure to download and place that folder in the parent folder
* cd scripts
* python [extract.py](extract.py)
	* picks each of the pickle file from ../pickleFiles/
	* Aided by [sessionIndex.csv](sessionIndex.csv) file and [gps_processed.csv](gps_processed.csv) file, the script generates sampled (10Hz) and interpolated CSV for each trip saved in [../tripCsvFiles](../tripCsvFiles) under the name "xxx_features_agg.csv"*
	* It also saves unsampled brake pressure data if required for future in the same folder under the format "xxx_unsampled_brake.csv"
 
* **now generate braking zones/labels**
* python [mergeBraking.py](mergeBraking.py)
	* The script takes each file of the name format *"xxx.features_agg.csv"* from the folder [../tripCsvFiles](../tripCsvFiles)
	* Calculates and appends braking zones from brake pressure data and saves the csv in the format *"xxx.features_brake_agg.csv"* in the same folder [../tripCsvFiles](../tripCsvFiles)
* So the final extracted trip files are in the format *xxx.features_brake_agg.csv"* in the folder [../tripCsvFiles](../tripCsvFiles)




#### Create PCA data csv files from the csv data: [You may use this script if you want to create PCA csv files for trips separately]
* cd scripts
* python [dataToPCA.py](scripts/dataToPCA.py)
	* Set the training and testing trips in [*sessionIndex.csv*](scripts/sessionIndex.csv) under the column *"train_or_test_for_demo"*
	* The number of PCA components can also be set in the file
	* The script takes files *"xxx_feature_brake_agg.csv"* from [../tripCsvFiles](tripCsvFiles) and generates corresponding pca files stored in the folder [../tripCsvFiles/pca](/tripCsvFiles/pca) under the format *"xxx_feature_brake_agg_pca.csv"*