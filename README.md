# EasyVIZ Demo:

A demonstration of two potential use cases for our EasyVIZ package.

### Use During Training
demo.py showcases using our Tensorflow Callbacks to create and save our layer visualizations, maximum mean filter activation visualizations, and our 2D-Entropy calculations. It also demonstrates how users can display the visualizations during traing, save them to disk, or log them to Tensorboard. We've chosen a very small model and dataset for this demo, so results can be seen instantly. Given the size of the model and dataset, however, the results may not appear very impressive.


### Use On Pretrained Models
demo2.py showcases how one could use EasyVIZ to explore pretrained models by visualizing maximum mean filter activations and computing the 2D-Entropy of these features. 
The demo uses a pretrained VGG16 model, but it is compatible with any Tensorflow model with accessible layers. Pretrained models from Tensorflow's tf.keras.applications section should work as long as the preprocessing function is provided, but models from Tensorflow's Model Zoo, for example, will not work because they are repacked as one-layer models. 

## Requirements:

tensorflow 2.x

tensorflow_addons

matplotlib

numpy

scipy

pathlib

imageio
