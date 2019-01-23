##################################################################################################################################
#
# Different network architectures to be selected.
# Adapted from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d/alexnet.py
#
#
##################################################################################################################################


import tensorflow as tf
import numpy as np
from functools import reduce


class VR_net(object):

	def __init__(self, x, mean_values, keep_prob, num_classes, filter_std=1e-2, bias_ini=0.0,  deep='vgg11b'):
		"""
		Inputs:
		- x: tf.placeholder, for the input images
		- mean_values: list, mean RGB values of the images of training
		- keep_prob: tf.placeholder, for the dropout rate
		- num_classes: int, number of classes of the new dataset
		- filter_std: initialization of the standard deviation of the weights
		- bias_ini: initialization of the biases
		- deep: network architecture to be created
		"""
		# Parse input arguments
		self.X = x
		self.MEAN_VALUES = mean_values
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob
		self.FILTER_STD = filter_std
		self.BIAS_INI = bias_ini
		self.WEIGHTS_PATH = 'vgg16_weights.npz'	# just implemented for the vgg16 network

		# Call the create function to build the computational graph of the selected network
		self.create(deep)

	def create(self, deep):

		self.parameters = []

		# Preprocess: zero-mean input
		self.x_pre = preproc(self.X, self.MEAN_VALUES, int(self.X.get_shape()[-1]))

		# Compute the computational graph according to the selected network
		if deep == 'vgg11b':
			# 1st Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv1_1, kernel, biases = conv(self.x_pre, 3, 3, 16, 1, 1, name='conv1_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv1_2, kernel, biases = conv(self.conv1_1, 3, 3, 16, 1, 1, name='conv1_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, name='pool1')

			# 2nd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv2_1, kernel, biases = conv(self.pool1, 3, 3, 32, 1, 1, name='conv2_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv2_2, kernel, biases = conv(self.conv2_1, 3, 3, 32, 1, 1, name='conv2_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, name='pool2')

			# 3rd Layer: Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Conv (with ReLu) -> Pool
			self.conv3_1, kernel, biases = conv(self.pool2, 3, 3, 64, 1, 1, name='conv3_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv3_2, kernel, biases = conv(self.conv3_1, 3, 3, 64, 1, 1, name='conv3_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv3_3, kernel, biases = conv(self.conv3_2, 3, 3, 64, 1, 1, name='conv3_3', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.conv3_4, kernel, biases = conv(self.conv3_3, 3, 3, 64, 1, 1, name='conv3_4', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]

			self.pool3 = max_pool(self.conv3_4, 2, 2, 2, 2, name='pool3')

			fc_in = self.pool3
			fc_out_size = 1024

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

		if deep != 'all-cnn':
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

		if deep == 'all-cnn':

			if self.KEEP_PROB != 1:
				self.dropout0 = dropout(self.x_pre, 0.8)
			else:
				self.dropout0 = dropout(self.x_pre, 1)

			# 1st Layer: Conv (with ReLu) -> Conv (with ReLu)
			self.conv1_1, kernel, biases = conv(self.dropout0, 3, 3, 96, 1, 1, name='conv1_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer = tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			self.conv1_2, kernel, biases = conv(self.conv1_1, 3, 3, 96, 1, 1, name='conv1_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			# 2nd Layer: Conv (with ReLu) stride 2
			self.conv2_1, kernel, biases = conv(self.conv1_2, 3, 3, 96, 2, 2, name='conv2_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			self.dropout2 = dropout(self.conv2_1, self.KEEP_PROB)

			# 3rd Layer: Conv (with ReLu) -> Conv (with ReLu)
			self.conv3_1, kernel, biases = conv(self.dropout2, 3, 3, 192, 1, 1, name='conv3_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			self.conv3_2, kernel, biases = conv(self.conv3_1, 3, 3, 192, 1, 1, name='conv3_2', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			# 4th Layer: Conv (with ReLu) stride 2
			self.conv4_1, kernel, biases = conv(self.conv3_2, 3, 3, 192, 2, 2, name='conv4_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			self.dropout4 = dropout(self.conv4_1, self.KEEP_PROB)

			# 5th Layer: Conv (with ReLu)
			self.conv5_1, kernel, biases = conv(self.dropout4, 3, 3, 192, 1, 1, name='conv5_1', padding='VALID', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			# 6th Layer: Conv (with ReLu)
			self.conv6_1, kernel, biases = conv(self.conv5_1, 1, 1, 192, 1, 1, name='conv6_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			# 7th Layer: Conv (with ReLu)
			self.conv7_1, kernel, biases = conv(self.conv6_1, 1, 1, self.NUM_CLASSES, 1, 1, name='conv7_1', filter_std=self.FILTER_STD, bias_ini=self.BIAS_INI)
			self.parameters += [kernel, biases]
			self.regularizer += tf.nn.l2_loss(kernel)
			self.regularizer += tf.nn.l2_loss(biases)

			# Global averaging over 6x6 spatial dimensions
			self.fc8 = tf.nn.avg_pool(self.conv7_1, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='VALID', name='fc8')
			self.fc8 = self.fc8[:, 0, 0, :]




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
Predefine all necessary layers for the net
"""

def preproc(x, mean_values, input_channels):
	'''Preprocessing step before apply any layer'''
	# zero-mean input
	with tf.variable_scope('preprocess') as scope:
		mean = tf.constant(mean_values, dtype=tf.float32, shape=[1, 1, 1, input_channels], name='img_mean')
		return x - mean

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, filter_std, bias_ini=0.0, padding='SAME', groups=1):
	'''Create a convolutional layer'''
	# Get number of input channels
	input_channels = int(x.get_shape()[-1])


	with tf.variable_scope(name) as scope:
		# Create tf variables for the weights and biases of the conv layer
		kernel = tf.Variable(tf.truncated_normal([filter_height, filter_width, input_channels/groups, num_filters], dtype=tf.float32, stddev=filter_std), name='weights')
		biases = tf.Variable(tf.constant(bias_ini, shape=[num_filters], dtype=tf.float32), trainable=True, name='biases')

		if groups == 1:
			# Apply the convolution
			conv = tf.nn.conv2d(x, kernel, strides=[1, stride_y, stride_x, 1], padding=padding)

		else:
			convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

			# Split input and weights and convolve them separately
			input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
			weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=kernel)
			output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

			# Concat the convolved output together again
			conv = tf.concat(axis=3, values=output_groups)

		# Add biases
		#out = tf.nn.bias_add(conv, biases)
		out = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

		# Apply relu function
		relu = tf.nn.relu(out, name=scope.name)

		return relu, kernel, biases


def fc(x, num_in, num_out, name, filter_std, bias_ini=0.0, relu=True):
	'''Create a fully connected layer'''
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
	'''Max-pooling'''
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides=[1, stride_y, stride_x, 1],
                        padding=padding, name=name)


def dropout(x, keep_prob):
	'''Dropout, keep_prob equal to 1 if we want to be removed.'''
	return tf.nn.dropout(x, keep_prob)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)