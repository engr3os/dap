#########

# The format of the files in the csv is such that
* Each column consists of 13 features - 9 face features (bin and angular histogram), 2 lane features (lane on left and lane on right), 1 intersection feature (15meters from intersection), 1 speed
* 7 consecutive columns make one sample. (Since the sampling rate is 0.8 sec and the duration of sample is 5 secs)