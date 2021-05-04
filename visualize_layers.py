# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:56:50 2021

@author: Robert Ronan
"""

import sys, os
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import math

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# # Rescale the images from [0,255] to the [0.0,1.0] range.
# #x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
# x_test = x_test /255
# x_train = x_train / 255

# x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
# x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
# y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


class VisualizeVggFeatures(tf.keras.callbacks.Callback):
    def __init__(self,
              sample_image=None):
      self.sample_image=sample_image


    def on_epoch_end(self, epoch, logs={}):

        model = self.model

        num_layers = 6
        if self.sample_image is None:
            randomNoise = np.random.rand(1,32,32,3)
            randomNoise = randomNoise * 255
        else:
            randomNoise = self.sample_image
            if len(randomNoise.shape) == 3:
                randomNoise = tf.expand_dims(randomNoise, axis=0)

        layer_outputs = [layer.output for layer in model.layers[1:num_layers+1]] # Gathers the outputs of the layers we want
        activation_model = Model(inputs=model.input, outputs=layer_outputs) # Isolates the model layers from our model
        activations = activation_model.predict(randomNoise) # Returns a list of five Numpy arrays: one array per layer activation

        images_per_row = 16

        layer_names = []
        for layer in model.layers[:num_layers]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot


        for layer_name, layer_activation in zip(layer_names, activations): # Iterates over every layer
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            output_size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            layer_vis = np.zeros((output_size * n_cols, images_per_row * output_size))


            for col in range(n_cols):
                for row in range(images_per_row):
                    feature = layer_activation[0, :, :, col * images_per_row + row]
                    # Scale and transform the activation for display
                    feature -= feature.mean() # Subtract the mean
                    feature /= feature.std() # Normalize

                    # Don't allow the intensity values to be too large (max 200... over 200 is harsh to look at)
                    feature *= 50
                    feature += 150
                    feature = np.clip(feature, 0, 255).astype('uint8')
                    # displays a panel of
                    layer_vis[col * output_size : (col + 1) * output_size,
                                row * output_size : (row + 1) * output_size] = feature
            scale = 1. / output_size
            plt.figure(figsize=(scale * layer_vis.shape[1],
                                scale * layer_vis.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(layer_vis, aspect='auto', cmap='plasma')
            plt.show()

