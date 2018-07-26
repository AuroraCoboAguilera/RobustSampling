##################################################################################################################################
#
# VGG16 based net for NMINST DATASET
# Adapted from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d/alexnet.py
#
# Aurora Cobo Aguilera
#
# Creation date: 02/07/2018
# Modification date: 03/07/2018
#
##################################################################################################################################


import tensorflow as tf
import numpy as np
from functools import reduce


class VGG_based_net(object):

	def __init__(self, x, mean_values, keep_prob, num_classes, filter_std=1e-2, bias_ini=0.0,  deep='vgg11b'):
		"""
		Inputs:
		- x: tf.placeholder, for the input images
		- mean_values: list, mean RGB values of the images of training
		- keep_prob: tf.placeholder, for the dropout rate
		- num_classes: int, number of classes of the new dataset
		- weights_path: path string, path to the pretrained weights, (if vgg16_weights.npz is not in the same folder)
		"""
		# Parse input arguments
		self.X = x
		self.MEAN_VALUES = mean_values
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob
		self.FILTER_STD = filter_std
		self.BIAS_INI = bias_ini
		self.WEIGHTS_PATH = 'vgg16_weights.npz'

		# Call the create function to build the computational graph of VGG based net
		self.create(deep)

	def create(self, deep):

		self.parameters = []

		# Preprocess: zero-mean input
		self.x_pre = preproc(self.X, self.MEAN_VALUES, int(self.X.get_shape()[-1]))

		if deep == 'vgg11b':
			# 1st Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv1_1, kernel, biases = conv(self.x_pre, 3, 3, 16, 1, 1, name='conv1_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv1_2, kernel, biases = conv(self.conv1_1, 3, 3, 16, 1, 1, name='conv1_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name='pool1')

			# 2nd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv2_1, kernel, biases = conv(self.pool1, 3, 3, 32, 1, 1, name='conv2_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv2_2, kernel, biases = conv(self.conv2_1, 3, 3, 32, 1, 1, name='conv2_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name='pool2')

			# 3rd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv3_1, kernel, biases = conv(self.pool2, 3, 3, 64, 1, 1, name='conv3_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv3_2, kernel, biases = conv(self.conv3_1, 3, 3, 64, 1, 1, name='conv3_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv3_3, kernel, biases = conv(self.conv3_2, 3, 3, 64, 1, 1, name='conv3_3', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv3_4, kernel, biases = conv(self.conv3_3, 3, 3, 64, 1, 1, name='conv3_4', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool3 = max_pool(self.conv3_4, 2, 2, 2, 2, name='pool3')

			fc_in = self.pool3
			fc_out_size = 1024

		if deep == 'vgg10':
			# 1st Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv1_1, kernel, biases = conv(self.x_pre, 3, 3, 32, 1, 1, name='conv1_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv1_2, kernel, biases = conv(self.conv1_1, 3, 3, 32, 1, 1, name='conv1_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name='pool1')

			# 2nd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv2_1, kernel, biases = conv(self.pool1, 3, 3, 64, 1, 1, name='conv2_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv2_2, kernel, biases = conv(self.conv2_1, 3, 3, 64, 1, 1, name='conv2_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name='pool2')

			# 3rd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv3_1, kernel, biases = conv(self.pool2, 3, 3, 128, 1, 1, name='conv3_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv3_2, kernel, biases = conv(self.conv3_1, 3, 3, 128, 1, 1, name='conv3_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.pool3 = max_pool(self.conv3_2, 2, 2, 2, 2, name='pool3')

			# 4th Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv4_1, kernel, biases = conv(self.pool3, 3, 3, 256, 1, 1, name='conv4_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]


			self.pool4 = max_pool(self.conv4_1, 2, 2, 2, 2, name='pool4')

			fc_in = self.pool4
			fc_out_size = 256

		if deep == 'vgg7':
			# 1st Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv1_1, kernel, biases = conv(self.x_pre, 3, 3, 16, 1, 1, name='conv1_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool1 = max_pool(self.conv1_1, 2, 2, 2, 2, name='pool1')


			# 2nd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv2_1, kernel, biases = conv(self.pool1, 3, 3, 32, 1, 1, name='conv2_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool2 = max_pool(self.conv2_1, 2, 2, 2, 2, name='pool2')


			# 3rd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv3_1, kernel, biases = conv(self.pool2, 3, 3, 64, 1, 1, name='conv3_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv3_2, kernel, biases = conv(self.conv3_1, 3, 3, 64, 1, 1, name='conv3_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]


			self.pool3 = max_pool(self.conv3_1, 2, 2, 2, 2, name='pool3')

			fc_in = self.pool3
			fc_out_size = 1024

		elif deep == 'vgg11':

			# 1st Layer: Conv (with ReLu) -> Pool
			self.conv1_1, kernel, biases = conv(self.x_pre, 3, 3, 64, 1, 1, name='conv1_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool1 = max_pool(self.conv1_1, 2, 2, 2, 2, name='pool1')

			# 2nd Layer: Conv (with ReLu) -> Pool
			self.conv2_1, kernel, biases = conv(self.pool1, 3, 3, 128, 1, 1, name='conv2_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool2 = max_pool(self.conv2_1, 2, 2, 2, 2, name='pool2')

			# 3rd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv3_1, kernel, biases = conv(self.pool2, 3, 3, 256, 1, 1, name='conv3_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv3_2, kernel, biases = conv(self.conv3_1, 3, 3, 256, 1, 1, name='conv3_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool3 = max_pool(self.conv3_2, 2, 2, 2, 2, name='pool3')

			# 4th Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv4_1, kernel, biases = conv(self.pool3, 3, 3, 512, 1, 1, name='conv4_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv4_2, kernel, biases = conv(self.conv4_1, 3, 3, 512, 1, 1, name='conv4_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool4 = max_pool(self.conv4_2, 2, 2, 2, 2, name='pool4')

			# 5th Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv5_1, kernel, biases = conv(self.pool4, 3, 3, 512, 1, 1, name='conv5_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv5_2, kernel, biases = conv(self.conv5_1, 3, 3, 512, 1, 1, name='conv5_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool5 = max_pool(self.conv5_2, 2, 2, 2, 2, name='pool4')

			fc_in = self.pool5
			fc_out_size = 4096

		elif deep == 'vgg16':
			# 1st Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv1_1, kernel, biases = conv(self.x_pre, 3, 3, 64, 1, 1, name='conv1_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv1_2, kernel, biases = conv(self.conv1_1, 3, 3, 64, 1, 1, name='conv1_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name='pool1')

			# 2nd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv2_1, kernel, biases = conv(self.pool1, 3, 3, 128, 1, 1, name='conv2_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv2_2, kernel, biases = conv(self.conv2_1, 3, 3, 128, 1, 1, name='conv2_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name='pool2')

			# 3rd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv3_1, kernel, biases = conv(self.pool2, 3, 3, 256, 1, 1, name='conv3_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv3_2, kernel, biases = conv(self.conv3_1, 3, 3, 256, 1, 1, name='conv3_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv3_3, kernel, biases = conv(self.conv3_2, 3, 3, 256, 1, 1, name='conv3_3', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool3 = max_pool(self.conv3_3, 2, 2, 2, 2, name='pool3')

			# 4th Layer: Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv4_1, kernel, biases = conv(self.pool3, 3, 3, 512, 1, 1, name='conv4_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv4_2, kernel, biases = conv(self.conv4_1, 3, 3, 512, 1, 1, name='conv4_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv4_3, kernel, biases = conv(self.conv4_2, 3, 3, 512, 1, 1, name='conv4_3', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool4 = max_pool(self.conv4_3, 2, 2, 2, 2, name='pool4')

			# 5th Layer: Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv5_1, kernel, biases = conv(self.pool4, 3, 3, 512, 1, 1, name='conv5_1', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv5_2, kernel, biases = conv(self.conv5_1, 3, 3, 512, 1, 1, name='conv5_2', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.conv5_3, kernel, biases = conv(self.conv5_2, 3, 3, 512, 1, 1, name='conv5_3', filter_std=self.FILTER_STD)
			self.parameters += [kernel, biases]

			self.pool5 = max_pool(self.conv5_3, 2, 2, 2, 2, name='pool4')

			fc_in = self.pool5
			fc_out_size = 4096

		# 6th Layer: Flatten -> FC (w ReLu) -> Dropout
		shape = int(np.prod(fc_in.get_shape()[1:]))
		flattened = tf.reshape(fc_in, [-1, shape])
		self.fc6, weights, biases = fc(flattened, shape, fc_out_size, name='fc6', filter_std=self.FILTER_STD,  relu=True)
		self.parameters += [weights, biases]

		self.dropout6 = dropout(self.fc6, self.KEEP_PROB)

		# 7th Layer: FC (w ReLu) -> Dropout
		self.fc7, weights, biases = fc(self.dropout6, fc_out_size, fc_out_size, name='fc7', filter_std=self.FILTER_STD, relu=True)
		self.parameters += [weights, biases]

		self.dropout7 = dropout(self.fc7, self.KEEP_PROB)

		# 8th Layer: FC (without ReLu) and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
		self.fc8, weights, biases = fc(self.dropout7, fc_out_size, self.NUM_CLASSES, name='fc8', filter_std=self.FILTER_STD, relu=False)
		self.parameters += [weights, biases]


	def load_initial_weights(self, session, path, skip_layer, no_layer):

		weights = np.load(path)
		keys = sorted(weights.keys())
		backw = 0
		for i, k in enumerate(keys):
			if k not in skip_layer:
				if k not in no_layer:
					print(i, i-backw, k, np.shape(weights[k]))
					session.run(self.parameters[i-backw].assign(weights[k]))
				else:
					backw += 1


	def save_npy(self, sess, npy_path="./vgg11-save.npy"):
		assert isinstance(sess, tf.Session)

		data_dict = {}

		for var in list(self.parameters):
			var_out = sess.run(var)
			data_dict[var.name] = var_out

		np.save(npy_path, data_dict)
		print(("file saved", npy_path))
		return npy_path


	def get_var_count(self):
		count = 0
		for v in list(self.parameters):
			count += reduce(lambda x, y: x * y, v.get_shape().as_list())
		return count


	def load_initial_weights(self, session, skip_layers=[]):

		weights = np.load(self.WEIGHTS_PATH)
		keys = sorted(weights.keys())
		for i, k in enumerate(keys):
			if k not in skip_layers:
				print(i, k, np.shape(weights[k]))
				session.run(self.parameters[i].assign(weights[k]))


"""
Predefine all necessary layer for the VGG11 net
"""

def preproc(x, mean_values, input_channels):
	# zero-mean input
	with tf.variable_scope('preprocess') as scope:
		mean = tf.constant(mean_values, dtype=tf.float32, shape=[1, 1, 1, input_channels], name='img_mean')
		return x - mean

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, filter_std, bias_ini=0.0, padding='SAME'):
	# Get number of input channels
	input_channels = int(x.get_shape()[-1])


	with tf.variable_scope(name) as scope:
		# Create tf variables for the weights and biases of the conv layer
		kernel = tf.Variable(tf.truncated_normal([filter_height, filter_width, input_channels, num_filters], dtype=tf.float32, stddev=filter_std), name='weights')
		biases = tf.Variable(tf.constant(bias_ini, shape=[num_filters], dtype=tf.float32), trainable=True, name='biases')

		# Apply the convolution
		conv = tf.nn.conv2d(x, kernel, strides=[1, stride_y, stride_x, 1], padding=padding)

		# Add biases
		out = tf.nn.bias_add(conv, biases)#tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

		# Apply relu function
		relu = tf.nn.relu(out, name=scope.name)

		return relu, kernel, biases


def fc(x, num_in, num_out, name, filter_std, bias_ini=0.0, relu=True):
	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases
		weights = tf.Variable(tf.truncated_normal([num_in, num_out], dtype=tf.float32, stddev=filter_std), trainable=True, name='weights')
		biases = tf.Variable(tf.constant(bias_ini, shape=[num_out], dtype=tf.float32), trainable=True, name='biases')

		# Matrix multiply weights and inputs and add bias
		act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

		if relu == True:
			# Apply ReLu non linearity
			relu = tf.nn.relu(act)
			return relu, weights, biases
		else:
			return act, weights, biases


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides=[1, stride_y, stride_x, 1],
                        padding=padding, name=name)


def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)
