# Introduction

This project is meant to be used on extracted Polysync data currently store on 10.0.0.7 (artemis)
    
/mnt/disk1/polysync/polysync_processed

# Usage
Clone the project to your "HOME" directory

Also ensure that the CAN file has been converted to json format.

If not the script will attempt the conversion, so please ensure that the conversion program "can_converter-0.9.jar" is also in your HOME directory.

## For pure data_extraction accross the extracted files of each driving trip

$ data_extractor.py trip_foldername

Output: 
* a pickle file with all extracted data with each time modality stored as pandas dataframe object

## For synchronized data

$ data_sync.py trip_foldername

Output: 
* a pickle file with all non-zero features synchronized and combined into a single padas dataframe object
* a csv version of the above

All features are synchronized to one of the cameras at 30fps. The features are resampled using forward fill to replace missing data.

This may be changed in the future.

## For data labeling
$ data_label.py trip_folderhome

Output: 
* Data visualization window where all data can be visualized and replayed for labeling
* data_label.cvs file that contains all hand annotated labels

The labels are stored per timestamp of the synchronized data   
