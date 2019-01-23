##################################################################################################################################
#
# Training of the different networks and dataset. It has multiple arguments to be defined acording to the dataset,
# network and configuration of them.
# Adapted from:
# https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d/validate_alexnet_on_imagenet.ipynb
#
#
##################################################################################################################################

import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from datetime import datetime
from VR_net import VR_net
from tensorflow.examples.tutorials.mnist import input_data
from datagenerator import ImageDataGenerator
from datagenerator import convert_labels_to_one_hot
from scipy.misc import imresize
import scipy.io as sio
import pickle



# Applying gcn followed by whitening
def global_contrast_normalize(X, scale=1., min_divisor=1e-8):
	X = X - X.mean(axis=1)[:, np.newaxis]

	normalizers = np.sqrt((X ** 2).sum(axis=1)) / scale
	normalizers[normalizers < min_divisor] = 1.

	X /= normalizers[:, np.newaxis]

	return X


def compute_zca_transform(imgs, filter_bias=0.1):
	meanX = np.mean(imgs, 0)

	covX = np.cov(imgs.T)

	D, E = np.linalg.eigh(covX + filter_bias * np.eye(covX.shape[0], covX.shape[1]))

	assert not np.isnan(D).any()
	assert not np.isnan(E).any()
	assert D.min() > 0

	D = D ** -0.5

	W = np.dot(E, np.dot(np.diag(D), E.T))
	return meanX, W


def zca_whiten(train, test, cache=None):
	if cache and os.path.isfile(cache):
		with open(cache, 'rb') as f:
			(meanX, W) = pickle.load(f)
	else:
		meanX, W = compute_zca_transform(train)

		with open(cache, 'wb') as f:
			pickle.dump((meanX, W), f, 2)

	train_w = np.dot(train - meanX, W)
	test_w = np.dot(test - meanX, W)

	return train_w, test_w



def build_parser():
	parser = ArgumentParser()
	parser.add_argument('--num_epochs', default=150, help="Number of training epochs (default: 150)", type=int)
	parser.add_argument('--batch_size', default=128, help="Number of samples that belong to a mini-batch (default: 128)", type=int)
	parser.add_argument('--learning_rate', default=0.001, help="Learning rate for the optimizer (default: 0.001) If 0, it is used an adaptive learning rate "
															   "that starts with the value of 0.01 and decreases at certain numbers of epochs. More precisely, "
															   "divides in log scale the total number of epochs in 3 stages, and then change to 0.001 after the "
															   "first stage and to 0.0001 after the second stage.", type=float)
	parser.add_argument('--use_lr_schedule', default=False, help="Use learning rate schedule. It consists on dividing the learning rate by 10 every of the three "
							 "epochs from the lr_schecule argument(default: false).", type=bool)
	parser.add_argument('--lr_schedule', default=[200, 250, 300], help="Learning rate schedule for the adaptive scheme. They are the epochs where the learning "
																	   "rate decreases 10 times (default: [200, 250, 300])", type=list)
	parser.add_argument('--weight_decay', default=0.001, help="Weight decay that can be used in all-conv architecture (default: 0.001)", type=float)
	parser.add_argument('--dropout_rate', default=0.5, help="Configuration of the regularization by dropout. In order to remove it, choose its value to 1.0 "
															"(default: 0.5)", type=float)
	parser.add_argument('--filter_std', default=0.1, help="Standard deviation for the normal initialization of the parameters of the filters in the network, "
														  "both convolutional filters and the ones of the fully connected layers (default: 0.1)", type=float)
	parser.add_argument('--bias_ini', default=0.0, help="Constant initialization for the biases of the filters in all the layers (default: 0.0)", type=float)
	parser.add_argument('--repetition_percentage', default=0.25, help="Rate of samples that are repeated in one of the robust methods (default: 0.25)", type=float)
	parser.add_argument('--save_dir', default=None, help="Extension to the base for the name directory to save the results. The base name is the concatenation of the "
														 "dataset with the network with the method. For example mnist_vgg11b_basic would be the base name for the training "
														 "of the mnist dataset with the vgg11b network and the baseline method (without robust approach) (default: None)")
	parser.add_argument('--gpu_num', default=0, help="Number of the GPU to be used by the NVIDIA commands. It is as it was executed the python script after the command "
													 "configuration CUDA_VISIBLE_DEVICES = 0 (default: 0)")
	parser.add_argument("--model_name", default="basic", help="It is the approach to be used in the selection of the samples in the mini-batch. There are 5 choices: "
															  "basic / batchsel1 / batchsel2 / batchsel1prob / batchsel2prob, where the first one is the baseline, the "
															  "method number 1 is the VR-M, the method 2 is the VR-E and the prob refers to the probabilistic approach "
															  "(default: basic)")
	parser.add_argument("--augmentation_type", default="none", help="It has two options 'none' and 'flip'. With 'flip', the images suffer an horizontal flip with "
																	"probability 0.5 when calling the next mini-batch in the training.  (default: none)")
	parser.add_argument("--load_exist_model", default=False, help="If True, it is continued a previous training of the same network by providing the checkpoint files "
																  "(default: false)", type=bool)
	parser.add_argument("--load_dir", default=None,help="Extension to the directory to load the parameters of the pre-trained model. The default extension is none and the "
														"base name is the concatenation of the dataset with the network with the method (default: dataset_deep_method)")
	parser.add_argument("--dataset", default="mnist", help="Name of the dataset to be trained. It is needed the data in a folder with the name of the dataset in uppercase inside"
														   " the folder 'data'. mnist / cifar10 / cifar100 / tinyImagenet / svhn.(default: mnist)")
	parser.add_argument("--deep", default="vgg11b", help="Selection of the neural network architecture as defined in VR_net; vgg11b / vgg11 / vgg16 / vgg7 / alexnet / "
														 "all-cnn / alexnet_cifar (default: vgg11b)")
	parser.add_argument("--rescale", default=None, help="New size of the images if it is desired them to be resized;  (default: None)", type=int)
	parser.add_argument("--preprocess", default=False, help="whitened and contrast normalized  (default: False) Cifar-10, Cifar-100",type=bool)
	parser.add_argument("--finetune", default=False, help="Boolean to finetune the network vgg16 pretrained on ImageNet (only with vgg16 deep)", type=bool)
	parser.add_argument("--layers", default=3, help="Number of the last layers to be trained if it is chosen to do a finetuning the vgg16 (default: 3) 1 / 2 / 3 / 4 / 5", type=int)
	parser.add_argument("--subset", default=None, help="Maximum number of samples per class in the training set. Only implemented with cifar10 (default: None) ", type=int)
	parser.add_argument("--train_samples", default=None, help="Number of training samples (default: None, which is 55000 and 10000 test samples in MNIST for example) ", type=int)
	parser.add_argument("--regularizer", default=True, help="Use the regularizer of the ALL-CNN architecture. Only implemented with all-cnn deep. (default: True)", type=bool)

	return parser

parser = build_parser()
FLAGS = parser.parse_args()
print("\nParameters:")
for attr, value in sorted(vars(FLAGS).items()):
	print("{}={}".format(attr.upper(), value))
print("")
print("========== save_dir is tmp!!! ==========")

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_num)

# Learning params
learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
# Network params
dropout_rate = FLAGS.dropout_rate
filter_std = FLAGS.filter_std
bias_ini = FLAGS.bias_ini
# Configuration params
deep = FLAGS.deep
method = FLAGS.model_name
dataset = FLAGS.dataset
repetitionPercentage = FLAGS.repetition_percentage
weight_decay = FLAGS.weight_decay

# Finetune
if FLAGS.layers == 1:
	noload_parameters = ['fc8_W', 'fc8_b']
	train_layers = ['fc8']
elif FLAGS.layers == 2:
	noload_parameters = ['fc8_W', 'fc8_b', 'fc7_W', 'fc7_b']
	train_layers = ['fc8', 'fc7']
elif FLAGS.layers == 3:
	noload_parameters = ['fc8_W', 'fc8_b', 'fc7_W', 'fc7_b', 'fc6_W', 'fc6_b']
	train_layers = ['fc8', 'fc7', 'fc6']
elif FLAGS.layers == 4:
	noload_parameters = ['fc8_W', 'fc8_b', 'fc7_W', 'fc7_b', 'fc6_W', 'fc6_b', 'conv5_2_b', 'conv5_2_W']
	train_layers = ['fc8', 'fc7', 'fc6', 'conv5_2']
elif FLAGS.layers == 5:
	noload_parameters = ['fc8_W', 'fc8_b', 'fc7_W', 'fc7_b', 'fc6_W', 'fc6_b', 'conv5_2_b', 'conv5_2_W', 'conv5_1_b', 'conv5_1_W']
	train_layers = ['fc8', 'fc7', 'fc6', 'conv5_2', 'conv5_1']


if FLAGS.learning_rate == 0.0:
	learning_rate = 0.01
	print('Adapting learning rate: ', learning_rate)

if FLAGS.load_dir == None:
	load_path = "tmp/" + dataset + '_' + deep + '_' + method + "/checkp"
else:
	load_path = "tmp/" + dataset + '_' + deep + '_' + method + FLAGS.load_dir + "/checkp"

if FLAGS.augmentation_type == "none":
	horizontal_flip = False
elif FLAGS.augmentation_type == "flip":
	horizontal_flip = True

# Dataset
if dataset == "mnist":
	mnist = input_data.read_data_sets('data/MNIST', one_hot=True)
	im_size = 28
	im_channels = 1
	n_classes = 10
	mean_values = [0.13070031]

	train_images = mnist.train.images
	train_labels = mnist.train.labels

	test_images = mnist.test.images
	test_labels = mnist.test.labels

	train_images_re = np.reshape(train_images, [np.shape(train_images)[0], im_size, im_size, im_channels])
	test_images_re = np.reshape(test_images, [np.shape(test_images)[0], im_size, im_size, im_channels])

	if FLAGS.train_samples != None:
		if FLAGS.train_samples < 65000:
			num_train_s = FLAGS.train_samples
			num_test_s = 65000 - FLAGS.train_samples
			print('Number of train samples: ', str(num_train_s))
			print('Number of test samples: ', str(num_test_s))

			total_images_re = np.concatenate((train_images_re, test_images_re))
			train_images_re = total_images_re[:num_train_s, :, :, :]
			test_images_re = total_images_re[num_train_s:, :, :, :]
			total_labels = np.concatenate((train_labels, test_labels))
			train_labels = total_labels[:num_train_s, :]
			test_labels = total_labels[num_train_s:, :]


elif dataset == "cifar10":
	im_size = 32
	n_classes = 10
	im_channels = 3
	if not(FLAGS.preprocess):
		mean_values = [0.5623179, 0.5562303, 0.52719558]
		import cifar10

		cifar10.data_path = "data/CIFAR-10/"

		#cifar10.maybe_download_and_extract()	# Error en python 2

		class_names = cifar10.load_class_names()
		print(class_names)

		images_train, cls_train, labels_train = cifar10.load_training_data()
		images_test, cls_test, labels_test = cifar10.load_test_data()

		if FLAGS.train_samples != None:
			if FLAGS.train_samples < 60000:
				num_train_s = FLAGS.train_samples
				num_test_s = 60000 - FLAGS.train_samples
				print('Number of train samples: ', str(num_train_s))
				print('Number of test samples: ', str(num_test_s))

				total_images_re = np.concatenate((images_train, images_test))
				images_train = total_images_re[:num_train_s, :, :, :]
				images_test = total_images_re[num_train_s:, :, :, :]
				total_labels = np.concatenate((labels_train, labels_test))
				labels_train = total_labels[:num_train_s, :]
				labels_test = total_labels[num_train_s:, :]
				total_cls = np.concatenate((cls_train, cls_test))
				cls_train = total_cls[:num_train_s]
				cls_test = total_cls[num_train_s:]

		if FLAGS.subset != None:
			images_train, cls_train, labels_train = cifar10.load_training_data_subset(images_train, cls_train, FLAGS.subset)

	else:
		mean_values = [0.0, 0.0, 0.0]
		batchdir = "data/CIFAR-10/cifar-10-batches-py"

		train_batches = [os.path.join(batchdir, "data_batch_" + str(i)) for i in range(1, 6)]

		Xlist, ylist = [], []
		for batch in train_batches:
			with open(batch, 'rb') as f:
				d = pickle.load(f, encoding='bytes')
				Xlist.append(d[b'data'])
				ylist.append(d[b'labels'])

		X_train = np.vstack(Xlist)
		y_train = np.vstack(ylist)

		with open(os.path.join(batchdir, "test_batch"), 'rb') as f:
			d = pickle.load(f, encoding='bytes')
			X_test, y_test = d[b'data'], d[b'labels']

		y_train = np.array(y_train).reshape(-1, 1)
		y_test = np.array(y_test).reshape(-1, 1)

		norm_scale = 55.0
		X_train = global_contrast_normalize(X_train, scale=norm_scale)
		X_test = global_contrast_normalize(X_test, scale=norm_scale)

		zca_cache = os.path.join(os.getcwd(), 'cifar-10-zca-cache.pkl')
		X_train, X_test = zca_whiten(X_train, X_test, cache=zca_cache)

		# Reformatting data as images
		images_train = X_train.reshape((X_train.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1))
		images_test = X_test.reshape((X_test.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1))

		# Covert to one hot
		labels_train = convert_labels_to_one_hot(y_train, n_classes)
		labels_test = convert_labels_to_one_hot(y_test, n_classes)
		labels_train = labels_train[:, 0, :]
		labels_test = labels_test[:, 0, :]

		if FLAGS.train_samples != None:
			if FLAGS.train_samples < 60000:
				num_train_s = FLAGS.train_samples
				num_test_s = 60000 - FLAGS.train_samples
				print('Number of train samples: ', str(num_train_s))
				print('Number of test samples: ', str(num_test_s))

				total_images_re = np.concatenate((images_train, images_test))
				images_train = total_images_re[:num_train_s, :, :, :]
				images_test = total_images_re[num_train_s:, :, :, :]
				total_labels = np.concatenate((labels_train, labels_test))
				labels_train = total_labels[:num_train_s, :]
				labels_test = total_labels[num_train_s:, :]

	if FLAGS.rescale != None:
		print('previous: ', images_train.shape)
		images_train = np.array(list(map(lambda img: imresize(img, [int(FLAGS.rescale), int(FLAGS.rescale), im_channels]), images_train)))
		images_test = np.array(list(map(lambda img: imresize(img, [int(FLAGS.rescale), int(FLAGS.rescale), im_channels]), images_test)))
		print('after: ', images_train.shape)
		im_size = int(FLAGS.rescale)


elif dataset == "svhn":

	data_path = "data/SVHN/"

	train_data = sio.loadmat(data_path + 'train_32x32.mat')
	test_data = sio.loadmat(data_path + 'test_32x32.mat')

	# access to the dict
	x_train = train_data['X']/255.0
	y_train = train_data['y'].flatten()
	y_train[y_train == 10] = 0
	x_test = test_data['X']/255.0
	y_test = test_data['y'].flatten()
	y_test[y_test == 10] = 0
	# classes from 1 to 10 (0 coded to 10)


	if FLAGS.train_samples != None:
		if FLAGS.train_samples < 99289:
			num_train_s = FLAGS.train_samples
			num_test_s = 99289 - FLAGS.train_samples
			print('Number of train samples: ', str(num_train_s))
			print('Number of test samples: ', str(num_test_s))

			total_images = np.concatenate((x_train, x_test), axis=3)
			x_train = total_images[:, :, :, :num_train_s]
			x_test = total_images[:, :, :, num_train_s:]
			total_labels = np.concatenate((y_train, y_test))
			y_train = total_labels[:num_train_s]
			y_test = total_labels[num_train_s:]


	im_size = 32
	n_classes = 10
	im_channels = 3
	mean_values = [0.4376821,  0.4437697,  0.47280442]

	# Covert to one hot
	y_train = convert_labels_to_one_hot(y_train, n_classes)
	y_test = convert_labels_to_one_hot(y_test, n_classes)




def train():

	"""
	Configuration settings
	"""

	# How often we want to write the tf.summary data to disk
	display_step = 1

	# Path for tf.summary.FileWriter and to store model checkpoints
	if FLAGS.save_dir == None:
		filewriter_path = "tmp/" + dataset + '_' + deep + '_' + method
		checkpoint_path = "tmp/" + dataset + '_' + deep + '_' + method + "/checkp"
	else:
		filewriter_path = "tmp/" + dataset + '_' + deep + '_' + method + FLAGS.save_dir
		checkpoint_path = "tmp/" + dataset + '_' + deep + '_' + method + FLAGS.save_dir + "/checkp"


	# Initalize the data generator seperately for the training and validation set
	if FLAGS.dataset == "mnist":
		train_generator = ImageDataGenerator(train_images_re, train_labels, horizontal_flip=horizontal_flip, shuffle=True, scale_size=(im_size, im_size), nchannels=im_channels, nb_classes=n_classes)
		val_generator = ImageDataGenerator(test_images_re, test_labels, horizontal_flip=False, shuffle=False, scale_size=(im_size, im_size), nchannels=im_channels, nb_classes=n_classes)
	elif FLAGS.dataset == "cifar10":
		train_generator = ImageDataGenerator(images_train, labels_train, horizontal_flip=horizontal_flip, shuffle=True, scale_size=(im_size, im_size), nchannels=im_channels,nb_classes=n_classes)
		val_generator = ImageDataGenerator(images_test, labels_test, horizontal_flip=False, shuffle=False, scale_size=(im_size, im_size), nchannels=im_channels, nb_classes=n_classes)
	elif FLAGS.dataset == "svhn":
		train_generator = ImageDataGenerator(x_train, y_train, horizontal_flip=horizontal_flip, shuffle=True, scale_size=(im_size, im_size), nchannels=im_channels,nb_classes=n_classes, inverse=True)
		val_generator = ImageDataGenerator(x_test, y_test, horizontal_flip=False, shuffle=False, scale_size=(im_size, im_size), nchannels=im_channels, nb_classes=n_classes, inverse=True)


	print('Training set', train_generator.data_size)
	print('Validation set', val_generator.data_size)

	# Get the number of training/validation steps per epoch
	train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
	val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

	print("Training batches per epoch: ", train_batches_per_epoch)
	print("Validation batches per epoch: ", val_batches_per_epoch)


	# Create parent path if it doesn't exist
	if not os.path.isdir("tmp"): os.mkdir("tmp")
	if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
	if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


	# TF placeholder for graph input and output
	x = tf.placeholder(tf.float32, [None, im_size, im_size, im_channels], name='x_ph')
	y = tf.placeholder(tf.float32, [None, n_classes], name='y_ph')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob_ph')
	lr = tf.placeholder(tf.float32, name='learning_rate_ph')

	# Initialize model
	print('Mean value images:', mean_values)
	model = VR_net(x, mean_values, keep_prob, n_classes, filter_std=filter_std, bias_ini=bias_ini, deep=FLAGS.deep)

	print('Number of parameters: ', model.get_var_count())


	# Link variable to model output
	score = model.fc8

	# For testing after training
	softmax = tf.nn.softmax(score, name='softmax_score')

	# List of trainable variables of the layers we want to train
	if FLAGS.finetune:
		var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
	else:
		var_list = [v for v in tf.trainable_variables()]

	# Op for calculating the loss
	with tf.name_scope("cross_ent"):
		loss_n = tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y)
		if deep == 'all-cnn' and FLAGS.regularizer:
			loss = tf.reduce_mean(loss_n + weight_decay * model.regularizer)
		else:
			loss = tf.reduce_mean(loss_n)

	# Train op
	with tf.name_scope("train"):
		# Get gradients of all trainable variables
		gradients = tf.gradients(loss, var_list)
		gradients = list(zip(gradients, var_list))

		# Create optimizer and apply gradient descent to the trainable variables
		if deep == 'all-cnn':
			optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
		else:
			optimizer = tf.train.GradientDescentOptimizer(lr)
		train_op = optimizer.apply_gradients(grads_and_vars=gradients)

	# Add gradients to summary
	for gradient, var in gradients:
		tf.summary.histogram(var.name + '/gradient', gradient)

	# Add the variables we train to the summary
	for var in var_list:
		tf.summary.histogram(var.name, var)

	# Add the loss to summary
	tf.summary.scalar('cross_entropy', loss)

	# Evaluation op: Accuracy of the model
	with tf.name_scope("accuracy"):
		correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

	# Add the accuracy to the summary
	tf.summary.scalar('accuracy', accuracy)

	# Merge all summaries together
	merged_summary = tf.summary.merge_all()

	# Initialize the FileWriter
	writer = tf.summary.FileWriter(filewriter_path)

	# Initialize an saver for store model checkpoints (to save graph and parameters trained)
	saver = tf.train.Saver()
	# Tensorflow variables are only alive inside a session. So, you have to save the model inside a session by calling save method on saver object you just created.


	# Start Tensorflow session
	with tf.Session() as sess:
		# Initialize all variables
		sess.run(tf.global_variables_initializer())

		# Add the model graph to TensorBoard
		writer.add_graph(sess.graph)

		# Load parameters trained on Imagenet
		if FLAGS.finetune:
			model.load_initial_weights(sess, noload_parameters)

		learning_rate = FLAGS.learning_rate

		# Load the pretrained weights into the non-trainable layer
		epochsCompleted = 0
		if FLAGS.load_exist_model:
			saver.restore(sess, tf.train.latest_checkpoint(load_path))
			epochsCompleted = int(tf.train.latest_checkpoint(load_path).split("model_epoch")[1].split(".ckpt")[0])

		print("{} Start training...".format(datetime.now()))
		print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))
		test_acc_array=[]
		# Loop over number of epochs
		for epoch in range(num_epochs):
			if FLAGS.learning_rate == 0 and epoch == int(np.logspace(0, np.log10(num_epochs), 4)[0]):
				learning_rate = 0.01
				print('learning rate changed to: ', learning_rate)
			elif FLAGS.learning_rate == 0 and epoch == int(np.logspace(0, np.log10(num_epochs), 4)[1]):
				learning_rate = 0.001
				print('learning rate changed to: ', learning_rate)
			elif FLAGS.learning_rate == 0 and epoch == int(np.logspace(0, np.log10(num_epochs), 4)[2]):
				learning_rate = 0.0001
				print('learning rate changed to: ', learning_rate)

			if FLAGS.use_lr_schedule and epoch in [200, 250, 300]:
				learning_rate = learning_rate/10
				print('learning rate changed to: ', learning_rate)

			print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

			step = 1

			l_total = np.array([])
			xs_total = np.array([])
			ys_total = np.array([])

			while step < train_batches_per_epoch:

				# Get a batch of images and labels
				if method == "basic":
					batch_xs, batch_ys = train_generator.next_batch(batch_size)
					# And run the training op
					sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate, lr: learning_rate})

				elif method == "batchsel1":
					if step == 1:
						batch_xs, batch_ys = train_generator.next_batch_robust(batch_size, loss_previous=None, repetitionPercentage=0, first=True)
					else:
						batch_xs, batch_ys = train_generator.next_batch_robust(batch_size, loss_previous=l_n, repetitionPercentage=repetitionPercentage, first=False)
					# And run the training op
					_, l_n = sess.run([train_op, loss_n], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate, lr: learning_rate})

				elif method == "batchsel1prob":
					if step == 1:
						batch_xs, batch_ys = train_generator.next_batch_robust(batch_size, loss_previous=None, repetitionPercentage=0, first=True, prob=True)
					else:
						batch_xs, batch_ys = train_generator.next_batch_robust(batch_size, loss_previous=l_n, repetitionPercentage=repetitionPercentage, first=False, prob=True)
					# And run the training op
					_, l_n = sess.run([train_op, loss_n], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate, lr: learning_rate})

				elif method == "batchsel2" or method == "batchsel2prob":
					if epoch == 0:
						batch_xs, batch_ys = train_generator.next_batch_robust_2(batch_size, first=True)
					else:
						batch_xs, batch_ys = train_generator.next_batch_robust_2(batch_size, first=False)
					_, l_n = sess.run([train_op, loss_n], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate, lr: learning_rate})
					if step == 1:
						xs_total = np.copy(batch_xs)
						ys_total = np.copy(batch_ys)
						l_total = np.copy(l_n)
					else:
						xs_total = np.concatenate((xs_total, batch_xs))
						ys_total = np.concatenate((ys_total, batch_ys))
						l_total = np.concatenate((l_total, l_n))

				# Generate summary with the current batch of data and write to file
				if step % display_step == 0:
					s = sess.run(merged_summary, feed_dict={x: batch_xs,
															y: batch_ys,
															keep_prob: 1., lr: learning_rate})
					writer.add_summary(s, epoch * train_batches_per_epoch + step)

				step += 1

			# Validate the model on the entire validation set
			print("{} Start validation".format(datetime.now()))
			test_acc = 0.
			test_count = 0
			for _ in range(val_batches_per_epoch):
				batch_tx, batch_ty = val_generator.next_batch(batch_size)
				acc = sess.run(accuracy, feed_dict={x: batch_tx,
													y: batch_ty,
													keep_prob: 1., lr: learning_rate})
				test_acc += acc
				test_count += 1
			test_acc /= test_count
			print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
			test_acc_array.append(test_acc)

			if method == "batchsel2":
				# Replace samples with a determined percentage
				train_generator.replace_data(l_total, repetitionPercentage)
			elif method == "batchsel2prob":
				# Replace samples with a determined percentage
				train_generator.replace_data(l_total, repetitionPercentage, prob=True)

			# Reset the file pointer of the image data generator
			val_generator.reset_pointer()
			train_generator.reset_pointer()

			print("{} Saving checkpoint of model...".format(datetime.now()))

			# save checkpoint of the model
			checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + epochsCompleted + 1) + '.ckpt')
			save_path = saver.save(sess, checkpoint_name, write_meta_graph=True)

			print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


			np.save(filewriter_path + "/ValAcc.npy", np.array(test_acc_array))

		count = train_generator.getCount()

		np.save(filewriter_path + "/count.npy", np.array(count))


if __name__ == '__main__':
	train()

