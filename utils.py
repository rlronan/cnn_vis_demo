# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:07:57 2021

@author: Robert Ronan
"""

#%% Imports
import tensorflow as tf
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import pathlib
import os
import math
import warnings
from tensorflow_addons.image import utils as img_utils
from tensorflow_addons.utils import keras_utils
from tensorflow_addons.utils.types import TensorLike
from typing import Optional, Union, List, Tuple

#%%
def sobel_filter(image):
    # convert to tensor and [0,1]
    image /= 255
    image = tf.image.sobel_edges(image)
    sobel_y = image[:, :, :, :, 0] # sobel in y-direction
    sobel_y = tf.clip_by_value(sobel_y / 4 + 0.5, 0, 1) # remap to [0,1]

    sobel_x = image[:, :, :, :, 1] # sobel in x-direction
    sobel_x = tf.clip_by_value(sobel_x / 4 + 0.5, 0, 1) # remap to [0,1]
    image = tf.clip_by_value(0.5*sobel_x + 0.5*sobel_y, 0, 1)
    image *= 255
    return image
#%%
def _pad(
    image: TensorLike,
    filter_shape: Union[List[int], Tuple[int]],
    mode: str = "CONSTANT",
    constant_values: TensorLike = 0,
) -> tf.Tensor:
    """Explicitly pad a 4-D image.
    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.
    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height
        and width of the 2-D filter.
      mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
    """
    if mode.upper() not in {"REFLECT", "CONSTANT", "SYMMETRIC"}:
        raise ValueError(
            'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
        )
    constant_values = tf.convert_to_tensor(constant_values, image.dtype)
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


def mean_filter2d(
    image: TensorLike,
    filter_shape: Union[List[int], Tuple[int], int] = [3, 3],
    padding: str = "REFLECT",
    constant_values: TensorLike = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Perform mean filtering on image(s).
    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D mean filter. Can be a single integer
        to specify the same value for all spatial dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "mean_filter2d"):
        image = tf.convert_to_tensor(image, name="image")
        original_ndims = img_utils.get_ndims(image)
        image = img_utils.to_4D_image(image)

        filter_shape = keras_utils.normalize_tuple(filter_shape, 2, "filter_shape")

        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.dtypes.cast(image, tf.dtypes.float32)

        # Explicitly pad the image
        image = _pad(image, filter_shape, mode=padding, constant_values=constant_values)

        # Filter of shape (filter_width, filter_height, in_channels, 1)
        # has the value of 1 for each element.
        area = tf.constant(filter_shape[0] * filter_shape[1], dtype=image.dtype)
        filter_shape += (tf.shape(image)[-1], 1)
        kernel = tf.ones(shape=filter_shape, dtype=image.dtype)

        output = tf.nn.depthwise_conv2d(
            image, kernel, strides=(1, 1, 1, 1), padding="VALID"
        )

        output /= area

        output = img_utils.from_4D_image(output, original_ndims)
        return tf.dtypes.cast(output, orig_dtype)

def grid_display(array, num_rows=None, num_columns=None):
    """

    Display a list of images as a grid.

    Args:
        array (numpy.ndarray): 4D Tensor (batch_size, height, width, channels)

    Returns:
        numpy.ndarray: 3D Tensor as concatenation of input images on a grid

    CODE FROM: https://github.com/sicara/tf-explain/blob/master/tf_explain/utils/display.py

    """
    assert(len(array.shape) == 4)

    if num_rows is not None and num_columns is not None:
        total_grid_size = num_rows * num_columns
        if total_grid_size < len(array):
            warnings.warn(
                Warning(
                    "Given values for num_rows and num_columns doesn't allow to display "
                    "all images. Values have been overrided to respect at least num_columns"
                )
            )
            num_rows = math.ceil(len(array) / num_columns)
    elif num_rows is not None:
        num_columns = math.ceil(len(array) / num_rows)
    elif num_columns is not None:
        num_rows = math.ceil(len(array) / num_columns)
    else:
        num_rows = math.ceil(math.sqrt(len(array)))
        num_columns = math.ceil(math.sqrt(len(array)))

    number_of_missing_elements = num_columns * num_rows - len(array)
    # We fill the array with np.zeros elements to obtain a perfect square
    array = np.append(
        array,
        np.zeros((number_of_missing_elements, *array[0].shape)).astype(array.dtype),
        axis=0,
    )

    grid = np.concatenate(
        [
            np.concatenate(
                array[index * num_columns : (index + 1) * num_columns], axis=1
            )
            for index in range(num_rows)
        ],
        axis=0,
    )

    return grid




#%% image smoothness
def image_smoothness(image, filename_save):
  """

  Compute the 'smoothness' of an image. Currently Unused.

  Args:
    image: Tensorflow tensor image with 4 dimensions in the format:
      (Batch Size, Height, Width, Channels) or an Tensorflow tensor with
      3 dimensions in the format: (Height, Width, Channels).
    filename_save: the full filename (including path from local directory
      to the save directory) to save the image smoothness information as.

  CODE (SLIGHTLY MODIFIED) FROM: https://stackoverflow.com/questions/24671901/does-there-exist-a-way-to-directly-figure-out-the-smoothness-of-a-digital-imag

  """
  grayscale = False

  # reshape to remove batch dimension
  if len(tf.shape(image)) == 4:
    # remove last batch and channel dimensions if image is grayscale
    if tf.shape(image)[-1] == 1:
      image = tf.reshape(image, tf.shape(image)[1:3])
      grayscale = True
    else: # remove batch dimension
      image = tf.reshape(image, tf.shape(image)[1:])
  # remove last dimension if image is grayscale
  elif len(tf.shape(image)) == 3 and tf.shape(image)[-1] == 1:
    image = tf.reshape(image, tf.shape(image)[:2])
    grayscale = True

  fig = plt.figure()
  ax1 = fig.add_subplot(221)  # top left
  ax1.axis('off')
  ax2 = fig.add_subplot(222)  # top right
  ax2.axis('off')

  ax3 = fig.add_subplot(223)  # bot left
  ax3.axis('off')

  ax4 = fig.add_subplot(224)  # bot right
  ax4.axis('off')

  v = np.absolute(ndi.filters.laplace(image))
  if not grayscale:
    v2 = np.average(v, axis=2) # Mixing the channels down
  else:
    v2 = v
  v3 = np.average(v)

  fig.suptitle("Image Smoothness: " + str(round(v3, 6)))
  ax1.imshow(image)
  ax2.imshow(v/ np.max(v))
  ax3.imshow(v2/ np.max(v2))
  ax4.imshow(v2/ np.max(v2) > 0.5)

  plt.savefig(filename_save, format="png")
  plt.close()

  return(v3)

#%% delentropy
def delentropy(image, filename_save, save_to_disk=True, show_plots=False):
    """

    Compute and save the 2D-entropy of an image.
    Also saves the historgram information. Saved as as a PNG.
    See: https://arxiv.org/abs/1609.01117.

    Args:
      image: Tensorflow tensor image with 4 dimensions in the format
        (Batch Size, Height, Width, Channels) or an Tensorflow tensor with
        3 dimensions in the format: (Height, Width, Channels).
      filename_save: the full filename (including path from local directory to
        the save directory) to save the image entropy and histogram information as.

    CODE (SLIGHTLY MODIFIED) FROM: https://gist.github.com/mxmlnkn/5a225478bc576044b035ad4a45c54f73

    """

    fig = plt.figure( figsize = ( 9,8 ) )

    # reshape to remove batch dimension
    if len(tf.shape(image)) == 4:
      # remove batch and chanel dimensions if image is grayscale
      if tf.shape(image)[-1] == 1:
        image = tf.reshape(image, tf.shape(image)[1:3])
      else: # just remove channel dimension
        image = tf.reshape(image, tf.shape(image)[1:])
    # remove last dimension if image is grayscale
    elif len(tf.shape(image)) == 3 and tf.shape(image)[-1] == 1:
      image = tf.reshape(image, tf.shape(image)[:2])


    image = image.numpy()

    #print("image type: ", image.dtype)
    if np.max(image) <= 1:
      image = (image*255).astype('uint8')

    # Using a 2x2 difference kernel [[-1,+1],[-1,+1]] results in artifacts!
    # In tests the deldensity seemed to follow a diagonal because of the
    # assymetry introduced by the backward/forward difference
    # the central difference correspond to a convolution kernel of
    # [[-1,0,1],[-1,0,1],[-1,0,1]] and its transposed, produces a symmetric
    # deldensity for random noise.
    if True:
        # see paper eq. (4)
        fx = ( image[:,2:] - image[:,:-2] )[1:-1,:]
        fy = ( image[2:,:] - image[:-2,:] )[:,1:-1]
    # else:
    #     # throw away last row, because it seems to show some artifacts which it shouldn't really
    #     # Cleaning this up does not seem to work
    #     kernelDiffY = np.array( [ [-1,-1],[1,1] ] )
    #     fx = signal.fftconvolve( image, kernelDiffY.T ).astype( image.dtype )[:-1,:-1]
    #     fy = signal.fftconvolve( image, kernelDiffY   ).astype( image.dtype )[:-1,:-1]
    #print( "fx in [{},{}], fy in [{},{}]".format( fx.min(), fx.max(), fy.min(), fy.max() ) )

    diffRange = np.max( [ np.abs( fx.min() ), np.abs( fx.max() ), np.abs( fy.min() ), np.abs( fy.max() ) ] )
    if diffRange >= 200   and diffRange <= 255  : diffRange = 255
    if diffRange >= 60000 and diffRange <= 65535: diffRange = 65535


    if diffRange == 0:
      diffRange = 0.25
      H = 0
      xedges = [-0.5, 0.5]
      yedges = [-0.5, 0.5]
      delDensity = np.zeros(shape=(1,1))
      gamma = 1
    else:
      # see paper eq. (17)
      # The bin edges must be integers, that's why the number of bins and range depends on each other
      nBins = min(1024, diffRange+1)#min( 1024, 2*diffRange+1 )
      if image.dtype == np.float:
          nBins = 1024

      #print( "Bins = {}, Range of Diff = {}".format( nBins, diffRange ) )
      # Centering the bins is necessary because else all value will lie on
      # the bin edges thereby leading to assymetric artifacts
      dbin = 0 if image.dtype == np.float else 0.5

      r = diffRange + dbin

      delDensity, xedges, yedges = np.histogram2d( fx.flatten(), fy.flatten(), bins = nBins, range = [ [-0.5,r], [-0.5,r] ] )#[ [-r,r], [-r,r] ] )
      if nBins == 2*diffRange+1:
          assert( xedges[1] - xedges[0] == 1.0 )
          assert( yedges[1] - yedges[0] == 1.0 )

      # Normalization for entropy calculation. np.sum( H ) should be ( imageWidth-1 )*( imageHeight-1 )
      # The -1 stems from the lost pixels when calculating the gradients with non-periodic boundary conditions
      #assert( np.product( np.array( image.shape ) - 1 ) == np.sum( delDensity ) )
      delDensity = delDensity / np.sum( delDensity ) # see paper eq. (17)
      delDensity = delDensity.T

      # "The entropy is a sum of terms of the form p log(p). When p=0 you instead use the limiting value (as p approaches 0 from above), which is 0."
      # The 0.5 factor is discussed in the paper chapter "4.3 Papoulis generalized sampling halves the delentropy"
      H = - 0.5 * np.sum( delDensity[ delDensity.nonzero() ] * np.log2( delDensity[ delDensity.nonzero() ] ) ) # see paper eq. (16)

      # gamma enhancements and inversion for better viewing pleasure
      delDensity = np.max(delDensity) - delDensity

      gamma = 1.
      delDensity = ( delDensity / np.max( delDensity ) )**gamma * np.max( delDensity )

    ax = [
        fig.add_subplot( 221, title = "Example image, H=" + str( np.round( H, 5 ) ) ),
        fig.add_subplot( 222, title = "x gradient of image (color range: [" +
                              #str( np.round( -diffRange, 3 ) ) + "," + str( np.round( diffRange, 3 ) ) + "])" ),
                              str(0 ) + "," + str( np.round( diffRange, 3 ) ) + "])" ),
        fig.add_subplot( 223, title = "y gradient of image (color range: [" +
                              #str( np.round( -diffRange, 3 ) ) + "," + str( np.round( diffRange, 3 ) ) + "])" ),
                              str( 0 ) + "," + str( np.round( diffRange, 3 ) ) + "])" ),
        fig.add_subplot( 224, title = "Histogram of gradient (gamma corr. " + str(gamma) + " )" )
    ]
    ax[0].imshow( image, cmap=plt.cm.gray, interpolation='bicubic' )
    ax[1].imshow( fx , cmap=plt.cm.gray, vmin = -diffRange, vmax = diffRange, interpolation='bicubic')#vmin = -diffRange, vmax = diffRange )
    ax[2].imshow( fy , cmap=plt.cm.gray, vmin = -diffRange, vmax = diffRange, interpolation='bicubic')# vmin = -diffRange, vmax = diffRange )
    if matplotlib.__version__ <= '3.3':
      ax[3].imshow( delDensity  , cmap=plt.cm.gray, vmin = 0, interpolation='bicubic', origin='low',
            extent = [ xedges[0], xedges[-1], yedges[0], yedges[-1] ] )
    else:
      ax[3].imshow( delDensity  , cmap=plt.cm.gray, vmin = 0, interpolation='bicubic', origin='lower',
            extent = [ xedges[0], xedges[-1], yedges[0], yedges[-1] ] )

    fig.tight_layout()
    fig.subplots_adjust( top = 0.92 )
    if show_plots:
      plt.show()
    if save_to_disk:
      plt.savefig(pathlib.Path(str(filename_save) + str(H) + '.png'), format="png")
    plt.close()
    return H


#%% normalize and cast images
def normalize_cast(img):
  """

  Cast image to an image tensor of type tf.uint8 in the range [0,255] for saving.

  Args:
    img: an image castable to a 'numpy array'.

  Returns: 'tf.tensor' of type 'tf.uint8' with values in [0,255]

  """
  # don't normalize, image, just rescale to [0,1] --> [0,255]
  assert(len(img.shape) == 4)
  img = img.numpy().astype(float)

  alpha = np.min(img)
  # Sends alpha to zero if positive or neg
  img -= alpha
  beta = np.max(img)
  if np.min(img) != beta:
    img /= beta

  #   img is in [0,1] rather than [0,1) so saturate=True
  img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)

  return img

#%% Get Convolutional Layers
def get_conv_layers(model):
  """

  Return `list` of `tuples` of the conv layer number, the layer name,
  and it's layer index in the model.

  Args:
    Model: a Tensorflow model.

  Example Usage:
    >>> conv_layers = get_conv_layer(tf.keras.applications.VGG16)
    >>> show_conv_layers(layers=conv_layers)
    Conv layer #, \t layer name, \t layer index in model
    0, \t\t block1_conv1, \t\t 0
    1, \t\t block1_conv2, \t\t 3
    2, \t\t block2_conv1, \t\t 7
    ...

  """
  conv_layers = []
  k = 0
  for i in range(len(model.layers)):
    layer = model.layers[i]
    try:
        kernel = layer.kernel_size
        conv_layers.append((k, layer.name, i))
        k += 1
    except AttributeError:
        continue
  return conv_layers

#%% Get Convolutional Layers
def show_conv_layers(model=None, layers=None):
  """

  Display the convolutional layer number, layer name, and layer index in the model.

  Args: (One of the folllowing)
    Model: a Tensorflow model.

    layers: the `list` returned by `get_conv_layers(model)`

  Example Usage:
    >>> conv_layers = get_conv_layer(tf.keras.applications.VGG16)
    >>> show_conv_layers(layers=conv_layers)
    Conv layer #, \t layer name, \t layer index in model
    0, \t\t block1_conv1, \t\t 0
    1, \t\t block1_conv2, \t\t 3
    2, \t\t block2_conv1, \t\t 7
    ...

  Returns: Nothing.

  """
  assert (model is not None) or (layers is not None)
  if layers is None:
    if type(model) == 'list':
      layers = model
    else:
      layers = get_conv_layers(model)
  print("\nconv layer #, \t layer name, \t layer index in model")
  for l in layers:
    print(l[0], '\t\t', l[1], '\t\t', l[2])
  print("\n")

#%% Reset model
def reset_weights(model):
  """

  Reinitializes the model weights in place. May not work more than once per session.

  Args:
    model: a Tensorflow `model` with accesible layers.

  Returns: Nothing.

  CODE FROM: https://github.com/keras-team/keras/issues/341

  """
  for layer in model.layers:
      if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
          reset_weights(layer) #apply function recursively
          continue

      #where are the initializers?
      if hasattr(layer, 'cell'):
          init_container = layer.cell
      else:
          init_container = layer

      for key, initializer in init_container.__dict__.items():
          if "initializer" not in key: #is this item an initializer?
                continue #if no, skip it

          # find the corresponding variable, like the kernel or the bias
          if key == 'recurrent_initializer': #special case check
              var = getattr(init_container, 'recurrent_kernel')
          else:
              var = getattr(init_container, key.replace("_initializer", ""))

          if var is not None:
            var.assign(initializer(var.shape, var.dtype))
          #use the initializer