
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
import pdb
from skimage import measure
from skimage.transform import resize


# ----------------------Read image using its path--------------------------------------------
# Default value is gray-scale, and image is read by YCbCr format as the paper said.
def imread(path, is_grayscale=True):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

# ----------------To scale down and up the original image, first we need to have no remainder while scaling operation---
# We need to find modulo of height (and width) and scale factor.
# Then, subtract the modulo from height (and width) of original image size.
# There should be no remainder even after scaling.
def modcrop(image, scale=3):

  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image
#--------------------------------------------------------------------------------------------------------------------------




#--------------Preprocess single image file-----------------------------------------------------------------------------
#(1) Read original image as YCbCr format (and grayscale as default)
#(2) Normalize
#(3) Apply image file with bicubic interpolation
# Args:
# path: file path of desired file
# input_: image applied bicubic interpolation (low-resolution)
# label_: image with original resolution (high-resolution)
def preprocess(path, scale=3):

  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)
  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.
  input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)
  return input_, label_

#--------------------------------------------------------------------------------------------------------------------------




#--------------------------Set the image hyper parameters----------------------------------------------------------------
c_dim = 1
input_size = 255


# ---------------------Define the model weights and biases---------------------------------------------------------------
# define the placeholders for inputs and outputs
# ------ Set the weight of three conv layers
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
}
biases = {
    'b1': tf.Variable(tf.zeros([64]), name='b1'),
    'b2': tf.Variable(tf.zeros([32]), name='b2'),
    'b3': tf.Variable(tf.zeros([1]), name='b3')
}
path = './image/butterfly_GT.bmp'
image = tf.cast(np.asarray(preprocess(path)), tf.float32)



# --------------------Define the model layers with three convolutional layers----------------------------------------
# ------ Compute feature maps of input low-resolution images
# conv1 layer with biases and relu : 64 filters with size 9 x 9
# ------ Compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# ------ Compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv1 = tf.nn.relu(tf.nn.conv2d(inputs, weights['w1'], strides=[1, 1, 1, 1], padding='SAME', name='conv1') + biases['b1'])
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b2'])
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b3']


#-------------------------- Load the pre-trained model file-------------------------------------------------------
model_path = './model/model.npy'
model = np.load(model_path, encoding='latin1').item()#np.load = np_load_old


#------------------ Add your code here: show the weights of model and try to visualise---------------------------------
# ------ Show the weights of model and try to visualise
# variabiles (w1, w2, w3)
print("PRE-TRAINED MODEL")
print(model)


# ------------Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file----------------
# launch a session
sess = tf.Session()
for key in weights.keys():
  sess.run(weights[key].assign(model[key]))
for key in biases.keys():
  sess.run(biases[key].assign(model[key]))


#------------------------- Read the test image--------------------------------------------------------------------
blurred_image, groundtruth_image = preprocess('./image/butterfly_GT.bmp')
# Run the model and get the SR image
# transform the input to 4-D tensor
# run the session
input_ = np.expand_dims(np.expand_dims(blurred_image, axis=0), axis=-1)
output_1 = sess.run(conv1, feed_dict={inputs: input_})
output_ = sess.run(conv3, feed_dict={inputs: input_})
out = np.reshape(output_,(255,255))

# ------------------Save the blurred and SR images and compute the psnr--------------------------------------
groundtruth = scipy.misc.imsave('groundtruth.jpg', groundtruth_image)
blurred = scipy.misc.imsave('blurred.jpg', blurred_image)
output = scipy.misc.imsave('SR.jpg',out)
psnr_output = measure.compare_psnr(groundtruth_image, out)
psnr_blurred = measure.compare_psnr(groundtruth_image, blurred_image)
print('Super Resolution - PSNR: ', psnr_output)
print('Bicubic/Blurred - PSNR: ', psnr_blurred)
