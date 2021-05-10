# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:50:00 2021

@author: Robert Ronan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from feature_vis import log_conv_features_callback
from feature_vis import *
from visualize_layers import VisualizeVggFeatures

#%% IMPORT DATA:
##
##
##
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
#x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
x_test = x_test /255
x_train = x_train / 255

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

#%% DEFINE LOGGING:
##
##
##
log_dir = pathlib.Path("./logs/run1/")
if not os.path.exists(log_dir.parent): os.mkdir(log_dir.parent)
if not os.path.exists(log_dir): os.mkdir(log_dir)


file_writer = tf.summary.create_file_writer("./logs/run1/")
file_writer.set_as_default()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/run1/", histogram_freq=1,
                                                      write_graph=False, write_images=False,
                                                      update_freq='epoch', profile_batch=0,
                                                      embeddings_freq=0, embeddings_metadata=None)
tf.summary.experimental.set_step(0)

##
#%% DEFINE MODEL:

inputs = tf.keras.Input(shape=(32,32,3), name='inputs') # do not include batch size

x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                           padding='same', name='conv2d_1', activation='relu')(inputs)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                           padding='same', name='conv2d_2', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = tf.keras.layers.Dropout(0.25, name='dropout')(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1,
                            padding='same', name='conv2d_3', activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1,
                            padding='same', name='conv2d_4', activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5, name='dropout2')(x)
outputs = tf.keras.layers.Dense(10, name='outputs')(x)

tf.keras.backend.clear_session()

model = tf.keras.Model(inputs=inputs, outputs=outputs,
                               name="model")

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #
              metrics=['accuracy'])

model.summary()
#%% SHOW CONV LAYERS:
# Show conv layers for callback:
print("Showing the convolutional layers of the model: ")

show_conv_layers(model=model)


#%% DEFINTE FEATURE CALLBACK:

feature_callback = log_conv_features_callback(
            log_dir=pathlib.Path("./logs/run1/"),
            update_freq='epoch',
            update_freq_val=1,
            layer_nums=[0,3],
            filter_indices=[0,1,2,3],
            iterations=60,
            resizes=12,
            resize_factor=1.2,
            preprocess_func=None,
            entropy=True,
            scale_early_layers=False)

#            save_to_disk=True,
#            tensorboard_log=True)

layer_feature_callback = VisualizeVggFeatures(sample_image=None)

#%% TRAIN MODEL
##
##
model.fit(x_train,
          y_train,
          batch_size=1024,
          epochs=100,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, feature_callback, layer_feature_callback])

cnn_results = model.evaluate(x_test, y_test)
