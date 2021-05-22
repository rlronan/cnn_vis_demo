# EasyVIZ Demo:
A demonstration of two potential use cases for our EasyVIZ package.

For the package implementation please see: ![EasyVIZ](https://github.com/rlronan/conv_vis)

<p float="left">
  <img src="/sample_images/block3_conv3_filter_6.75103142.png" width="410" />
  <img src="/sample_images/block4_conv1_filter_6.645063669.png" width="410" /> 
</p>


## Use During Training
demo.py showcases using our Tensorflow Callbacks to create and save our layer visualizations, maximum mean filter activation visualizations, and our 2D-Entropy calculations. It also demonstrates how users can display the visualizations during traing, save them to disk, or log them to Tensorboard. We've chosen a very small model and dataset for this demo, so results can be seen instantly. Given the size of the model and dataset, however, the results may not appear very impressive.


## Use On Pretrained Models
demo2.py showcases how one could use EasyVIZ to explore pretrained models by visualizing maximum mean filter activations and computing the 2D-Entropy of these features. 
The demo uses a pretrained VGG16 model, but it is compatible with any Tensorflow model with accessible layers. Pretrained models from Tensorflow's tf.keras.applications section should work as long as the preprocessing function is provided, but models from Tensorflow's Model Zoo, for example, will not work because they are repacked as one-layer models. 

#### 2D-Entropy Examples With VGG16
Filters from Block 2, Conv 1. They appear to be looking for pixel gradients in a constrained range.
<p float="left">
  <img src="/sample_images/block2_conv1_8_entropy7.596859865217067.png" width="410" />
  <img src="/sample_images/block2_conv1_9_entropy7.443763865100494.png" width="410" /> 
</p>


Filters from Block 2, Conv 1. We observe two edge detection filters, and note the histogram of gradients are entirely different from the first two examples.
<p float="left">
  <img src="/sample_images/block2_conv1_10_entropy7.369202489645265.png" width="410" />
  <img src="/sample_images/block2_conv1_11_entropy7.257914880508193.png" width="410" /> 
</p>


Filters from Block 3, Conv 1. While they both appear to be detecting edges in the direction from top left to bottom right, the second filter also shows detections in the orthogonal direction. 
<p float="left">
  <img src="/sample_images/block3_conv1_4_entropy7.067398583164607.png" width="410" />
  <img src="/sample_images/block3_conv1_5_entropy7.373630438611548.png" width="410" /> 
</p>

Filters from Block 3, Conv 1.
<p float="left">
  <img src="/sample_images/block3_conv1_6_entropy7.604249326478307.png" width="410" />
  <img src="/sample_images/block3_conv1_9_entropy7.769468999401262.png" width="410" /> 
</p>

Filters from Block 3, Conv 4 and Conv 1.
<p float="left">
  <img src="/sample_images/block3_conv4_27_entropy7.373313278392066.png" width="410" />
  <img src="/sample_images/block4_conv1_23_entropy6.7876039289021435.png" width="410" /> 
</p>

## Requirements:

tensorflow 2.x

tensorflow_addons compatible with the installed tensorflow version

scipy

pathlib

imageio

matplotlib

numpy

