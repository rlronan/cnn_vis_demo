# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:28:51 2021

@author: Robert Ronan
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import imageio
import warnings
from feature_vis import *


model = tf.keras.applications.VGG16()


conv_layers = get_conv_layers(model)
show_conv_layers(layers=conv_layers)
"""
conv layer #, 	 layer name, 	 layer index in model
0 		           block1_conv1 		 1
1 		           block1_conv2 		 2
2 		           block2_conv1 		 4
3 		           block2_conv2 		 5
4 		           block3_conv1 		 7
5 		           block3_conv2 		 8
6 		           block3_conv3 		 9
7 		           block4_conv1 		 11
8 		           block4_conv2 		 12
9 		           block4_conv3 		 13
10 		           block5_conv1 		 15
11 		           block5_conv2 		 16
12 		           block5_conv3 		 17
"""

layer_nums = [2,6,8,12]#list(range(13))

log_dir = pathlib.Path("./demo_model/")
if not os.path.exists(log_dir): os.mkdir(log_dir)
resizes = 10
iterations = 250
resize_factor = 1.2
#%%
log_conv_features(model=model, layer_nums=layer_nums,
              preprocess_func=tf.keras.applications.vgg16.preprocess_input,
              directory=log_dir,
              iterations=iterations,
              resizes=resizes,
              resize_factor=resize_factor
              )
