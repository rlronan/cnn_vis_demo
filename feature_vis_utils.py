# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:15:11 2021

@author: Robert Ronan
"""
#%% Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import imageio
from utils import *

#%% get_features
def get_features(model=None, layer_name = None, preprocess_func=None,
                filter_index=0, iterations=200, step_size=1, resizes=10,
                resize_factor=1.2, clip=True):
  """
  Generate visualizations of convolutional features by computing the maximum mean
  filter activation (the image that maximizes the mean filter activation).

  In principal, the generated image should be 'what' the
  filter responds to most strongly; i.e. the feature it looks for.

  Function parameters will effect the outcome.

  Args:
    model: A Tensorflow `model` with accessible convolutional layers. You can use
      custom models, or (pretrained) `Tf.keras.application models`, but
      Tensorflow Model Garden `models` will not work, because their layers are
      repacked into a single layer.
    layer_name: The name of layer to save the feature visualizations of.
    preprocess_func: EITHER `None` if the model accepts inputs in the [0,1] range,
      OR 'default' if the model accepts inputs in the [-1,1] range,
      OR '255' if the model accpets inputs in the [0,255] range (as floats),
      OR a preprocessing function/layer in line with the specifications of
      a tf.keras.applications.model_name.preprocess_input function.
      See: https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input.
    filter_indices: a `list` or `np.array` of indices of the filters to visualize.
      (default: [0,...15]).
    iterations: An `int` specifying the total number of gradient ascent iterations
      to use when producing the feature visualization. Typical ranges are 50-200.
    step size: An `int` controlling the step size of gradient ascent. Generally
      it is not necessary to modify this. Values in the range of 1-5 are reasonable.
    resizes: An `int` specifying the number of times to resize upwards, crop,
      then add noise when generating the filter feature visualizations.
      (This can help eliminate high-frequency noise, and improve image quality.
       However, to many resizes can effect entropy calculations and image
       quality as well).
    resize_factor: A 'float' specifying how much to resize the image by during
      resizing.
    clip: (default: `True`) A `Boolean` controlling whether to 'clip' pixel values in the lower
      1/8 of the range to 0 (or -1 for images in [-1,1]). This can reduce noise
      and improve the quality of feature visualizations.

  Returns:
    The convolutional filter feature (the iamge that maximizes the mean filter activation)

  Code originally from tf_explain, but extensively modified.
  see also: https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030

  """

  # Create a connection between the input and the target layer
  if (preprocess_func is None) or (preprocess_func == 'default') or (preprocess_func == '255'):
    submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])
    tf_var = True      # input images will be `tf.Variables`
  else:
    # tf.keras.appplication models require a preprocessing layer, and
    #they cannot use tensorflow variables, so tensorflow tensors will be used.
    model_in = model.inputs[0]
    processed = preprocess_func(model_in)
    submodel_temp = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])
    processed = submodel_temp(processed)
    submodel = tf.keras.models.Model([model.inputs[0]], [processed])
    tf_var = False     # input images will be `tf.tensors`, not `tf.Variables`


  # Create initial random noise input data
  assert len(model.input_shape) == 4
  assert (model.input_shape[-1] == 3 or model.input_shape[-1] == 1)
  input_shape =  model.input_shape
  base_scale = 4

  # Set starting image to be 1/4 the size of the standard input
  img_shape = (1, input_shape[1]//base_scale, input_shape[2]//base_scale, input_shape[3]) # (batch_size, height, width, channels)
  def final_size(img_shape, resizes, resize_factor):
    size = img_shape[1]
    for r in range(resizes):
      size = int(size*resize_factor)
    return size

  # if the image won't be sufficently enlarged/shrunk so that the
  # final image is larger/smaller than the input, reduce/increase the base image
  #   scaling from 1/4 --> 1/3.75 --> 1/3.5 --> --> 1
  #   scaling from 1/4 --> 1/4.25 --> 1/4.5 -->
  new_scale = base_scale
  while final_size(img_shape, resizes, resize_factor) >=  1.5*input_shape[1]:
    new_scale += 0.25
    img_shape = (1, int(input_shape[1]//new_scale), int(input_shape[2]//new_scale), input_shape[3])
  while final_size(img_shape, resizes, resize_factor) < input_shape[1]:
    new_scale -= 0.25
    img_shape = (1, int(input_shape[1]//new_scale), int(input_shape[2]//new_scale), input_shape[3])

  if preprocess_func is None:
    # create image in [0,255] then cast to float in [0,1]
    input_img_data        = np.uint8(np.random.uniform(117,137, img_shape))/255
    input_img_data        = tf.Variable(tf.cast(input_img_data, tf.float32))


  elif preprocess_func == 'default':
    # create image in [0,255] then cast to float in [-1,1]
    input_img_data        = np.uint8(np.random.randint(117,137, img_shape))/127.5 - 1
    input_img_data        = tf.Variable(tf.cast(input_img_data, tf.float32))

  else: # '255' or tf.keras.application.[model].preprocess_input() both asssume images in a [0,255]
    # create image in [0,255] then cast to float.
    input_img_data        = np.uint8(np.random.randint(117,137, img_shape))
    input_img_data        = tf.cast(input_img_data, tf.float32)

  tf.get_logger().setLevel('ERROR') # else warnings every iteration if a tf.tensor is used
  opt = tf.keras.optimizers.Adam(learning_rate=0.1)

  for resize in range(resizes+2):
    for it in range(iterations//(resizes+2)):
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(input_img_data)
        outputs = submodel(input_img_data, training=False) # evaluation mode

        # calculate loss as the average value of the filter
        # and negate the loss because we're doing gradient ascent.
        loss_value = -1*tf.reduce_mean(outputs[:, :, :, filter_index])

      # calculate the gradient with resp to the input image, rather than the model vars
      grads = tape.gradient(loss_value, input_img_data)
      normalized_grads = grads / (
        tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

      # update input image, not the model, with gradient ascent
      if tf_var:
        opt.apply_gradients(zip([normalized_grads], [input_img_data]))
      else:
        input_img_data = tf.Variable(input_img_data) # cannot use Adam.apply_gradient on tensors
        opt.apply_gradients(zip([normalized_grads], [input_img_data]))
        input_img_data = tf.convert_to_tensor(input_img_data)

      # exceot for images in [0,1], we first clip values in the lower 1/8th of
      # the image range up, then subtract down. for Im
      if clip and (it % 5 == 0):
        if (preprocess_func is None):         # image in [0,1]
          input_img_data.assign(tf.keras.activations.relu(input_img_data, threshold=0.125))
        elif (preprocess_func == 'default'):  # image in [-1,1]
          input_img_data.assign(tf.math.maximum(input_img_data, -0.75))
          input_img_data.assign_sub( 0.25 * tf.cast(tf.math.equal(input_img_data,-0.75), tf.float32))
        elif (preprocess_func == '255'):      # image in [0,255]
          input_img_data.assign(tf.keras.activations.relu(input_img_data, threshold=32))

          # input_img_data.assign(tf.math.maximum(input_img_data, 32))
          # input_img_data.assign_sub( 32 * tf.cast(tf.math.equal(input_img_data, 32), tf.float32))
        else:                                 # image in [0,255], and is not a tf.var
          input_img_data = tf.keras.activations.relu(input_img_data, threshold=32)

          # input_img_data = tf.math.maximum(input_img_data, 32)
          # input_img_data -= 32 * tf.cast(tf.math.equal(input_img_data, 32), tf.float32)
    if resize < resizes:
      if tf_var:
        # Resize and blur the image to reduce high-frequency noise.
        size = np.array(input_img_data.shape)*resize_factor
        input_img_data = tf.Variable(tf.image.resize(input_img_data, (int(size[1]), int(size[2])), method='bicubic'))
        if input_img_data.shape[1] >= 56:
          input_img_data.assign(mean_filter2d(input_img_data, filter_shape=(5,5)))
        else:
          input_img_data.assign(mean_filter2d(input_img_data, filter_shape=(3,3)))

      else:
        size = np.array(input_img_data.shape)*resize_factor
        input_img_data = tf.image.resize(input_img_data, (int(size[1]), int(size[2])), method='bicubic')
        if input_img_data.shape[1] >= 56:
          input_img_data = mean_filter2d(input_img_data, filter_shape=(5,5))
        else:
          input_img_data = mean_filter2d(input_img_data, filter_shape=(3,3))


  tf.get_logger().setLevel('INFO')
  return input_img_data

#%% save features
def save_features(model=None, layer_name=None, preprocess_func=None,
                  save_directory=None, filter_indices=None, iterations=200,
                  step_size=1, resizes=10, resize_factor=1.2, clip=True,
                  step=None, entropy=True, save_to_disk=True, tensorboard_log=True,
                  show_plots=False):
  """
  Save visualizations of features and their entropy for convolutional layers.

  Features are computed by generating an image that maximizes the
  mean activation of a filter. In principal, this image should be 'what' the
  filter responds to most strongly; i.e. the features it looks for.

  The entropy of the features is calculated as 2D-Entropy (AKA delentropy;
  see : https://arxiv.org/abs/1609.01117). The entropy values are somewhat
  variable to the parameters used to generate the feature visualizations.
  Values >= 7.3~7.5 typically, but not always, indicate features of pure noise.
  Values <= 2 typically, but not always, indicate nearly monochromatic features.

  Args:
    model: A Tensorflow `model` with accessible convolutional layers. You can use
      custom models, or (pretrained) `Tf.keras.application models`, but
      Tensorflow Model Garden `models` will not work, because their layers are
      repacked into a single layer.
    layer_name: The name of layer to save the feature visualizations of.
    preprocess_func: EITHER `None` if the model accepts inputs in the [0,1] range,
      OR 'default' if the model accepts inputs in the [-1,1] range,
      OR '255' if the model accpets inputs in the [0,255] range (as floats),
      OR a preprocessing function/layer in line with the specifications of
      a tf.keras.applications.model_name.preprocess_input function.
      See: https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input.
    save_directory: a pathlib.Path of the save directory for the logs.
      E.g. directory=pathlib.Path('./models/VGG16/1/')
    filter_indices: a `list` or `np.array` of indices of the filters to visualize.
      (default: [0,...15]).
    iterations: An `int` specifying the total number of gradient ascent iterations
      to use when producing the feature visualization. Typical ranges are 50-200.
    step size: An `int` controlling the step size of gradient ascent. Generally
      it is not necessary to modify this. Values in the range of 1-5 are reasonable.
    resizes: An `int` specifying the number of times to resize upwards, crop,
      then add noise when generating the filter feature visualizations.
      (This can help eliminate high-frequency noise, and improve image quality.
       However, to many resizes can effect entropy calculations and image
       quality as well).
    resize_factor: A 'float' specifying how much to resize the image by during
      resizing.
    clip: (default: `True`) A `Boolean` controlling whether to 'clip' pixel values in the lower
      1/8 of the range to 0 (or -1 for images in [-1,1]). This can reduce noise
      and improve the quality of feature visualizations.
    step: (Optional) An `int` that gives the step for tensorboard logging.
    entropy: (default: `True`) `Boolean` describing whether or not to compute
      and save the 2D-entropy of the convolutional features visualized.
    save_to_disk: (default: `True`) `Boolean` describing whether or not to save
      feature images to disk.
    tensorboard_log: (default: `True`) `Boolean` describing whether or not to
      log data to tensorboard.

  Returns:
    An image containing the feature visualizations in a grid.

  Raises:
    IOError: (but continues) if attempt to save entropy files fails.

  """
  images = []
  count = 0
#TODO: add entropy variance
  entropy_total = 0  # average entropy across all filters examined
  if filter_indices is None:
    filter_indices = np.random.choice(32, size=16, replace=False)

  for idx in filter_indices:
    # get the 'features'; i.e. the images that maximally activate the mean value of the filter
    image = get_features(model, layer_name=layer_name, preprocess_func=preprocess_func,
                        filter_index=idx, iterations=iterations, step_size=step_size,
                        resizes=resizes, resize_factor=resize_factor, clip=clip)
    image = normalize_cast(image)     # recast image to [0,255]
    if count != 0:
      images = np.concatenate((images, image), axis=0)
    else:
      images = image
    count += 1

    if entropy:
      save_str = layer_name + '_' + str(idx) + '_entropy' #.png
      entropy_value = delentropy(image, filename_save = save_directory /  save_str,
                                 save_to_disk=save_to_disk, show_plots=show_plots)
      entropy_total += entropy_value

  entropy_total /= len(filter_indices)
  if entropy:
    #try:
    summary_str = str(layer_name) + "_entropy"
    if tensorboard_log:
      tf.summary.scalar(summary_str, data=entropy_total, step=step)
    if save_to_disk:
      try:
        with open(save_directory  / 'entropy_log.txt.', 'a+') as logfile:
          logfile.write("Layer: {},  Entropy: {}\n".format(layer_name, entropy_total))
      except IOError as e:
        tf.print("Could not write to log file at {}. Error text: {}".format(save_directory / 'entropy_log.txt.', e))

      if (save_directory.parent / 'filter_log.txt').exists():
        try:
          with open(save_directory.parent / 'filter_log.txt', 'a+') as kernel_log:
            kernel_log.write("Layer: {},  Entropy: {}\n".format(layer_name, entropy_total))
        except IOError as e:
          tf.print("Could not write to log file at {}. Error text: {}".format(save_directory.parent / 'filter_log.txt', e))
      elif (save_directory / 'filter_log.txt').exists():
        try:
          with open(save_directory / 'filter_log.txt', 'a+') as kernel_log:
            kernel_log.write("Layer: {},  Entropy: {}\n".format(layer_name, entropy_total))
        except IOError as e:
          tf.print("Could not write to log file at {}. Error text: {}".format(save_directory / 'filter_log.txt', e))

    #tf.print("Layer: {},  Entropy: {}".format(layer_name, np.round(entropy_total, 7)))
  if len(images) > 1:
    images = grid_display(images)
  if len(images) == 1:
    images = images[0]
  if entropy:
    image_name = layer_name + '_filter_' + str(np.round(entropy_total,9)) + '.png'
  else:
    image_name = layer_name + '_filter.png'
  if tensorboard_log:
    summary_str = str(layer_name) + "Features"
    images_tb = tf.expand_dims(images, axis=0)
    tf.summary.image(name=summary_str, data=images_tb, step=step)
  if save_to_disk:
    imageio.imwrite(save_directory / image_name, images, format='png')
  # if show_plots:
  #   plt.imshow(tf.reshape(images, list(images.shape)[:2]), cmap='gray')
  return images
