#!/usr/bin/env python
import json
import sys
import re
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as an
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from PIL import Image
import os
from glob import glob
import numpy as np
import pdb
from os.path import expanduser

""" This script read in the pickle file and display images and make plots 
"""

sys.path.append(expanduser("~"))
from data_labeling_tool.data_sync import sync

def onClick(event):
    global pause
    pause ^= True
def press(event):
  global delta, press_delta
  #print('press', event.key)
  if event.key == 'right':
    delta = 0
    press_delta = 1
  if event.key == 'left':
    delta = 0
    press_delta = -1

def combine_imgs(img_files):
  imgs = [Image.open(img_file).rotate(180) if (type(img_file) in [str] and ind==rot_index)  else Image.open(img_file) if (type(img_file) in [str] and ind!=rot_index) else Image.open(img_files[valid_ind]) for ind, img_file in enumerate(list(img_files)) ]
  #min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs if np.sum(i.size)>0])[0][1]
  imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
  return Image.fromarray(imgs_comb)

if __name__ == "__main__":
  args = sys.argv[1:]
  file_name = args[0]
  if len(args) > 1: 
    labels = args[1:-1]
    labels_file = args[-1]
  else:
    labels_file = raw_input("Please enter the full path to labels csv file: ")
    labels = raw_input("Please enter your labels, space separated: ")
    labels = labels.split()
  try:
    label_data = pd.read_csv(labels_file)
    #label_data['timestamp'] = pd.to_numeric(label_data['timestamp'])#.convert_objects(convert_numeric=True)
    label_data['timestamp'] = pd.to_datetime(label_data['timestamp'], unit='ns')
    label_data.set_index('timestamp', drop=True, inplace=True)
  except:
    pass
    
plot_labels = ['ssa', 'pbrk', 'hv_accp', 'sp1', 'b_p', 'yr', 'latitude', 'longitude', 'id', 'age', 'object_class', 'sizeY', 'throttle_position','leftlane_valid', 'leftlane_confidence', 'leftlane_boundarytype', 'rightlane_valid', 'rightlane_confidence','rightlane_boundarytype', 'face_cam', 'hand_cam', 'outside_cam']
img_labels = ['face_cam', 'hand_cam', 'outside_cam']

attr = []
attr_count = []
for num, label in enumerate(labels):
  attr.append(raw_input("Please enter attributes for label "+label+" (still space separated): ").split())
  attr_count.append(len(attr[num]))
num_class = max(attr_count)

file_name = os.path.abspath(file_name)
if os.path.isfile(file_name):
  print "Processing", os.path.split(file_name)[1]
  try:
    data = pd.read_csv(f)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
    data.set_index('timestamp', drop=True, inplace=True)
  except:
    with open(file_name, 'rb') as f:
      data = pickle.load(f)
  os.chdir(os.path.dirname(file_name))
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
  print "Found an existing extracted csv file"
  print "Processing", os.path.split(file_name)[1]
  data = pd.read_csv(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv')
  data['timestamp'] = pd.to_numeric(data['timestamp'])#.convert_objects(convert_numeric=True)
  data['timestamp'] = pd.to_datetime(data['timestamp'], unit='us')
  data.set_index('timestamp', drop=True, inplace=True)
  os.chdir(file_name)
elif os.path.isfile(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.csv'):
  print "Found an existing extracted pickle file"
  print "Processing", os.path.split(file_name)[1]
  with open(file_name+'/'+os.path.split(file_name)[1]+'_extr_synced.pkl', 'rb') as f:
    data = pickle.load(f)
  os.chdir(file_name)
else:
  data = sync(file_name)
  os.chdir(file_name)
  
for item in plot_labels:
  if not item in data.columns:
    data[item] = 0

for item in img_labels:
  if not item in data.columns:
    img_labels.remove(item)
      
label_data_sync = label_data.reindex(index=data.index, method='ffill')
data = pd.concat([data, label_data_sync], axis=1)
try:
  start_play, stop_play = input("If you know the start and stop timestamps, \nplease enter timestamps quoted and comma separated: ")
  data = data.loc[pd.to_datetime(start_play, unit='us'):pd.to_datetime(stop_play, unit='us')+pd.to_timedelta(5, unit='s')]
  print "Start play: ", pd.to_datetime(start_play, unit='us'), "\nEnd play: ", pd.to_datetime(stop_play, unit='us')
except ValueError:
  data = data.loc[pd.to_datetime(start_play):pd.to_datetime(stop_play)+pd.to_timedelta(5, unit='s')]
  print "Start play: ", pd.to_datetime(start_play), "\n End play: ", pd.to_datetime(stop_play)
except:
  print "Invalid timestamp format, data is not changed"
  #pass

#pdb.set_trace()      
img_tstamp = [label+"_timestamp" for label in img_labels]
data_label = pd.DataFrame(columns=img_tstamp+labels, index=data.index)
data_label[img_tstamp] = data[img_labels].applymap(lambda x: str(x).split('/')[-1].split('.')[0]) 
pause = False

img_files = data.loc[data.index[0],img_labels]
rot_index = list(img_files.keys()).index('face_cam')
valid_imgs = [(ind, img) for ind, img in enumerate(list(img_files)) if str(img) != str(np.nan)]
valid_ind = valid_imgs[0][0]
min_shape = Image.open(valid_imgs[0][1]).size
#plt_fig = plt.figure()
plt_fig = plt.figure(figsize=(20, 11.3), dpi=96)
# Texts
ax = plt.subplot2grid((3,3), (1,2))
plt.setp(ax.get_xaxis(), visible=False)
plt.setp(ax.get_yaxis(), visible=False)
text = ax.text(0.05, 0.05, 'Detected object:', transform=ax.transAxes)
# Images
#ax1 = plt_fig.add_subplot(3,1,1)
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax1.set_anchor('W')
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.setp(ax1.get_yticklabels(), visible=False)
plt.setp(ax1.get_xaxis(), visible=False)
plt.setp(ax1.get_yaxis(), visible=False)
plt_image = ax1.imshow(combine_imgs(img_files))
button = Button(ax1, '')
button.on_clicked(onClick)
# steering angle
#ax2 = plt_fig.add_subplot(3,3,4)
ax2 = plt.subplot2grid((3,3), (1,0))
plt_line, = ax2.plot([], [], 'r')
ax2.set_ylim(min(data['ssa'].min(), 0), data['ssa'].max())
ax2.set_xlim(0, data.shape[0]-1)
ax2.set_ylabel('Steering angle')
# speed
ax3 = plt.subplot2grid((3,3), (1,1))
#plt_line1, = ax3.plot([], [], 'b', ms=1)
plt_line1, = ax3.plot([], [], 'b')
ax3.set_ylim(min(data['sp1'].min(), 0), data['sp1'].max())
ax3.set_xlim(0, data.shape[0]-1)
ax3.set_ylabel('Speed')
## Yaw rate
ax5 = plt.subplot2grid((3,3), (2,0))
plt_line3, = ax5.plot([], [], 'k', label='Predicted Braking')
plt_line6, = ax5.plot([], [], 'r', label='Actual Brake Press')
#ax5.set_ylim(min(data[['yr', 'pbrk']].min().min(), 0), data[['yr', 'pbrk']].max().max())
#ax5.set_xlim(0, data.shape[0]-1)
#ax5.set_ylabel('Yaw Rate & Break Press')
#plt.legend(handles=[plt_line3, plt_line6])
data['pbrk'] = 5*data['pbrk']
ax5.set_ylim(min(data[['pbrk', 'pred_label']].min().min(), 0), min(1.5, data[['pbrk', 'pred_label']].max().max()))
ax5.set_xlim(0, data.shape[0]-1)
ax5.set_ylabel('Breaking')
plt.legend(handles=[plt_line3, plt_line6])
## Throttle Position
ax6 = plt.subplot2grid((3,3), (2,1))
#plt_line4, = ax6.plot([], [], 'r', label='Throttle (OBD)')
plt_line5, = ax6.plot([], [], 'b', label='Throttle (CAN)')
plt_line2, = ax6.plot([], [], 'g', label= 'Parking')
#throt_diff = data['throttle_position'].max()-data['throttle_position'].min()
#throt_min = data['throttle_position'].min()
accel_diff = data['hv_accp'].max()-data['hv_accp'].min()
accel_min = data['hv_accp'].min()
ax6.set_ylim(0, 1.0)
ax6.set_xlim(0, data.shape[0]-1)
ax6.set_ylabel('Throttle and parking Position')
#plt.legend(handles=[plt_line4, plt_line5, plt_line2])
plt.legend(handles=[plt_line5, plt_line2])
## Log and lat
ax7 = plt.subplot2grid((3,3), (2,2))
plt_line7, = ax7.plot([], [], 'r')
ax7.set_ylim(data['latitude'].min(), data['latitude'].max())
ax7.set_xlim(data['longitude'].min(), data['longitude'].max())
ax7.set_ylabel('Latitude')
ax7.set_xlabel('Longitude')
rax = plt.axes([0.53+0.2, 0.71, 0.06, 0.19])
radio = RadioButtons(rax, labels, active=0)
#radio = CheckButtons(rax, labels[1:], [False]*len(labels[1:]))
rax1 = plt.axes([0.60+0.2, 0.71, 0.06, 0.19])
radio1 = RadioButtons(rax1, attr[0], active=0)
#radio1 = CheckButtons(rax1, range(num_class), [False]*num_class)

FFMpegWriter = an.writers['ffmpeg']
#FFMpegWriter = an.writers['imagemagick']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
#writer = FFMpegWriter(fps=30, metadata=metadata)
writer = FFMpegWriter(fps=10)

def updatefig(itergen):
  i = itergen
  global img_files
  img_files = data.loc[data.index[i],img_labels]
  plt.suptitle('Processing timestamp '+str((data.index[i].value)//10**3)+', remaining '+str(data.index[-1]-data.index[i]).split()[-1]+'/'+str(data.index[-1]-data.index[0]).split()[-1]+' to complete trip', fontsize=14)
  plt_image.set_array(combine_imgs(img_files))
  plt_line.set_data(range(i), data.loc[data.index[:i], 'ssa'].values)
  plt_line1.set_data(range(i), data.loc[data.index[:i], 'sp1'].values)
  plt_line2.set_data(range(i), data.loc[data.index[:i], 'b_p'].values)
  plt_line3.set_data(range(i), data.loc[data.index[:i], 'pred_label'].values)
  #plt_line4.set_data(range(i), (data.loc[data.index[:i], 'throttle_position'].values-throt_min)/throt_diff)
  plt_line5.set_data(range(i), (data.loc[data.index[:i], 'hv_accp'].values-accel_min)/accel_diff)
  plt_line6.set_data(range(i), data.loc[data.index[:i], 'pbrk'].values)
  plt_line7.set_data(data.loc[data.index[:i], 'longitude'].values, data.loc[data.index[:i], 'latitude'].values)
  text.set_text('Detected object: '+str(data.loc[data.index[i], 'object_class'])+'\n'+
  'Object id: '+str(data.loc[data.index[i], 'id'])+'\n'+
  'Object size: '+str(data.loc[data.index[i], 'sizeY'])+'\n'+
  'Object age: '+str(data.loc[data.index[i], 'age'])+'\n'+
  'Leftlane_valid: '+str(data.loc[data.index[i], 'leftlane_valid'])+'\n'+
  'Rightlane_valid: '+str(data.loc[data.index[i], 'rightlane_valid'])+'\n'+
  'Leftlane_confidence: '+str(data.loc[data.index[i], 'leftlane_confidence'])+'\n'+
  'Rightlane_confidence: '+str(data.loc[data.index[i], 'rightlane_confidence'])+'\n'+
  'Leftlane_boundarytype: '+str(data.loc[data.index[i], 'leftlane_boundarytype'])+'\n'+
  'Rightlane_boundarytype: '+str(data.loc[data.index[i], 'rightlane_boundarytype'])+'\n'
  ) 
    
  #return plt_image, plt_line, plt_line1, plt_line2, plt_line3, plt_line4, plt_line5, plt_line6, plt_line7,text, start, stop, data_label 
  return plt_image, plt_line, plt_line1, plt_line2, plt_line3, plt_line5, plt_line6, plt_line7,text, data_label 

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt_fig.canvas.mpl_connect('key_press_event', press)
#plt_fig.canvas.mpl_connect('button_press_event', onClick)
#plt_ani = an.FuncAnimation(plt_fig, updatefig, itergen, interval = 10, blit=False, repeat=False)
plt_ani = an.FuncAnimation(plt_fig, updatefig, frames=data.shape[0], interval = 20, blit=False, repeat=False) # blit false due to slider
#pylab.get_current_fig_manager().window.showMaximized()
plt_ani.save(labels[0]+".mp4", writer=writer)
#plt_ani.save("data_moview.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()