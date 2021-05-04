import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import imageio
import warnings
from utils import *
from feature_vis_utils import *

def log_conv_features(model=None, layer_nums=None, preprocess_func=None,
                directory="./models/", filter_indices = np.arange(16),
                iterations=200, step_size=1, resizes=10, resize_factor=1.2,
                clip=True, scale_early_layers=True, train_step=None, entropy=True,
                save_to_disk=True, tensorboard_log=True, show_plots=False):
  """
  Save visualizations and entropy of convolutional layer features.

  Features are visualized by computing an image that maximizes the mean
  activation of a filter.

  Args:
    model: A Tensorflow `model` with accessible convolutional layers. You can use
      custom models, or (pretrained) `Tf.keras.application models`,
      but Tensorflow Model Garden `models` will not work, because their
      layers are repacked into a single layer.
    layer_nums: A `list' of  `integers` specifying the conv layers to visualize.
      use `get_conv_layers(model)` or `show_conv_layers(model)` to get the conv
      layer numbers. E.g. layer_nums=[0,1,2,7,9,15,16] would visualize the
      0th, 1st, 2nd, 7th, 9th, 15th, and 16th, convolutional layers that
      appear in the model. These layers probably will not be the 0th, 1st, ...16th
      layers in the model, since it may contain normalizing, and activation layers.
    preprocess_func: EITHER `None` if the model accepts inputs in the [0,1] range,
      OR 'default' if the model accepts inputs in the [-1,1] range,
      OR '255' if the model accpets inputs in the [0,255] range (as floats),
      OR a preprocessing function/layer in line with the specifications of
      a tf.keras.applications.model_name.preprocess_input function.
      See: https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input.
    directory: a pathlib.Path of the save directory for the logs.
      E.g. directory=pathlib.Path('./models/VGG16/')
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
    sigma: A `float` givng the standard deviation of the gaussian bluring during
      resizing.
    clip: A `Boolean` controlling whether to 'clip' pixel values in the lower
      1/8 of the range to 0 (or -1 for images in [-1,1]). This can reduce noise
      and improve the quality of feature visualizations. (default: `True`).
    train_step: (Optional) An `int` specifying the training iteration step.
    entropy: A `Boolean` controlling whether the entropy of the visualized
      features is computed and saved.

  Returns:
    Nothing

  Example Usage:
    >>> model = tf.keras.applications.VGG16()
    >>> show_conv_layers(model=my_model)
    conv layer #, 	 layer name, 	 layer index in model
    0 		 block1_conv1 		 1
    1 		 block1_conv2 		 2
    2 		 block2_conv1 		 4
    .
    .
    .
    10 		 block5_conv1 		 15
    11 		 block5_conv2 		 16
    12 		 block5_conv3 		 17
    >>> log_conv_features(model, layer_nums=[0,1,6,5,11,12], preprocess_func=None,
                directory=pathlib.Path('./feature_logs/'), filter_indices=np.arange(16),
                iterations=200, step_size=1, resizes=10, resize_factor=1.2, entropy=True)

  """

  if not os.path.exists(directory):
    os.mkdir(directory)

  conv_layers = get_conv_layers(model)
  iterations_base = iterations
  resizes_base = resizes
  if layer_nums is None:
    warnings.warn("If you do not pass a list of ints specifying which conv layers \
                  you want logged, then every conv layer will be logged.\
                    For non-trivial models, this can be extremely time consuming.")
    layer_nums = list(range(len(conv_layers)))
  for conv_layer_index in layer_nums:
    layer_name = conv_layers[conv_layer_index][1]
    if scale_early_layers and (conv_layer_index < 4):
      iterations = iterations_base
      resizes = resizes_base // 4
    else:
      iterations = iterations_base
      resizes = resizes_base
    try:
      save_features(model=model, layer_name=layer_name, preprocess_func=preprocess_func,
                   save_directory=directory, filter_indices=filter_indices,
                   iterations=iterations, step_size=step_size, resizes=resizes,
                   resize_factor=resize_factor, clip=clip, step=train_step, entropy=entropy,
                   save_to_disk=save_to_disk, tensorboard_log=tensorboard_log,
                   show_plots=show_plots)
    except ValueError as e:
      print(e)

#%% log_conv_features_callback
class log_conv_features_callback(tf.keras.callbacks.Callback):
  """
  Tensorflow callback for `log_conv_features` function.

  Save visualizations and entropy of convolutional layer features.

  Features are visualized by computing an image that maximizes the mean
  activation of a filter.

  Args:
    log_dir: A `string` path of the save directory for the logs.
    update_freq: One of the`strings` 'epoch' or 'batch' declaring the frequency
      of log_conv_feature updates.
    update_freq_val: An `int` specifying frequency value for the updates.
    overwite: (default: False) A `boolean` of whether logs should be overwritten
      every update by writing to the same folder. If entropy is true, overwrite
      will write to one folder, but won't actually overwrite the old images.
    layer_nums: A `list' of  `integers` specifying the conv layers to visualize.
      use `get_conv_layers(model)` or `show_conv_layers(model)` to get the conv
      layer numbers. E.g. layer_nums=[0,1,2,7,9,15,16] would visualize the
      0th, 1st, 2nd, 7th, 9th, 15th, and 16th, convolutional layers that
      appear in the model. These layers probably will not be the 0th, 1st, ...16th
      layers in the model, since it may contain normalizing, and activation layers.
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
    clip: A `Boolean` controlling whether to 'clip' pixel values in the lower
      1/8 of the range to 0 (or -1 for images in [-1,1]). This can reduce noise
      and improve the quality of feature visualizations. (default: `True`).
    train_step: (Optional) An `int` specifying the training iteration step.
    entropy: A `Boolean` controlling whether the entropy of the visualized
      features is computed and saved.

  Returns:
    Nothing

  Example Usage:
    >>> model = tf.keras.applications.VGG16()
    >>> show_conv_layers(model=model)
    conv layer #, 	 layer name, 	 layer index in model
    0 		 block1_conv1 		 1
    1 		 block1_conv2 		 2
    2 		 block2_conv1 		 4
    .
    .
    .
    10 		 block5_conv1 		 15
    11 		 block5_conv2 		 16
    12 		 block5_conv3 		 17
    >>> feature_callback = log_conv_features_callback(
            log_dir=pathlib.Path('./feature_logs/'),
            layer_nums=[0,1,2,10,11,12],
            preprocess_func=tf.keras.applications.vgg16.preprocess_input,
            clip=True, entropy=True)
    >>> history = model.fit(train_dataset, epochs=20, validation_data=val_dataset,
                            callbacks=[feature_callback])

  """

  def __init__(self,
               log_dir='feature_logs',
               update_freq='epoch',
               update_freq_val=1,
               overwrite=False,
               layer_nums=[0,1,2,3],
               preprocess_func=None,
               filter_indices=np.arange(16),
               iterations=200,
               step_size=1,
               resizes=10,
               resize_factor=1.2,
               clip=True,
               scale_early_layers=True,
               train_step=None,
               entropy=True,
               save_to_disk=True,
               tensorboard_log=True,
               show_plots=False):
    super(log_conv_features_callback, self).__init__()
    self.log_dir            = pathlib.Path(log_dir)
    self.update_freq        = update_freq
    self.update_freq_val    = update_freq_val
    self.overwrite          = overwrite
    self.layer_nums         = layer_nums
    self.preprocess_func    = preprocess_func
    self.filter_indices     = filter_indices
    self.iterations         = iterations
    self.step_size          = step_size
    self.resizes            = resizes
    self.resize_factor      = resize_factor
    self.clip               = clip
    self.scale_early_layers = scale_early_layers
    self.train_step         = train_step
    self.entropy            = entropy
    self.save_to_disk       = save_to_disk
    self.tensorboard_log    = tensorboard_log
    self.show_plots         = show_plots
    self.file_idx           = 0
  def on_epoch_end(self, epoch, logs=None):
    if self.update_freq != 'epoch':
      return
    if (epoch == 0) or (epoch % self.update_freq_val != 0):
      return
    if (not self.overwrite) and self.save_to_disk:
      if epoch == 1:
        log_dir_temp = pathlib.Path(self.log_dir / str(epoch))
        if not os.path.exists(log_dir_temp): os.mkdir(log_dir_temp)
        self.log_dir = pathlib.Path(log_dir_temp)
      else:
        log_dir_temp = pathlib.Path(self.log_dir.parent / str(epoch))
        if not os.path.exists(log_dir_temp): os.mkdir(log_dir_temp)
        self.log_dir = pathlib.Path(log_dir_temp)
    log_conv_features(model=self.model, layer_nums=self.layer_nums,
                preprocess_func=self.preprocess_func, directory=self.log_dir,
                filter_indices=self.filter_indices, iterations=self.iterations,
                step_size=self.step_size, resizes=self.resizes,
                resize_factor=self.resize_factor, clip=self.clip,
                scale_early_layers = self.scale_early_layers,
                train_step=self.train_step, entropy=self.entropy,
                save_to_disk=self.save_to_disk, tensorboard_log=self.tensorboard_log,
                show_plots=self.show_plots
                )

  def on_batch_end(self, batch, logs=None):
    if self.update_freq != 'batch':
      return
    if (batch == 1) or (batch % self.update_freq_val != 0):
      return
    if (not self.overwrite) and self.save_to_disk:
      if self.file_idx == 0:
        log_dir_temp = self.log_dir / str(self.file_idx)
        if not os.path.exists(log_dir_temp): os.mkdir(log_dir_temp)
        self.log_dir = pathlib.Path(log_dir_temp)
        self.file_idx += 1
      else:
        log_dir_temp = self.log_dir.parent / str(self.file_idx)
        if not os.path.exists(log_dir_temp): os.mkdir(log_dir_temp)
        self.log_dir = pathlib.Path(log_dir_temp)
        self.file_idx += 1
    log_conv_features(model=self.model, layer_nums=self.layer_nums,
                preprocess_func=self.preprocess_func, directory=self.log_dir,
                filter_indices=self.filter_indices, iterations=self.iterations,
                step_size=self.step_size, resizes=self.resizes,
                resize_factor=self.resize_factor, clip=self.clip,
                scale_early_layers = self.scale_early_layers,
                train_step=self.train_step, entropy=self.entropy,
                save_to_disk=self.save_to_disk, tensorboard_log=self.tensorboard_log,
                show_plots=self.show_plots
                )

