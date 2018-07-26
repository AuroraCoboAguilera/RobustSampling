##################################################################################################################################
#
# training VGG16 net
# Adapted from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d/validate_alexnet_on_imagenet.ipynb
#
# Aurora Cobo Aguilera
#
# Creation date: 02/07/2018
# Modification date: 03/07/2018
#
##################################################################################################################################

import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from datetime import datetime
from VGGbased_net import VGG_based_net
from tensorflow.examples.tutorials.mnist import input_data
from datagenerator import ImageDataGenerator
from datagenerator import convert_labels_to_one_hot
from scipy.misc import imresize
import scipy.io as sio



def build_parser():
	parser = ArgumentParser()
	parser.add_argument('--num_epochs', default=150, help="Number of training epochs (default: 150)", type=int)
	parser.add_argument('--batch_size', default=128, help="Batch size (default: 128)", type=int)
	parser.add_argument('--learning_rate', default=0.001, help="Learning rate for optimizer (default: 0.001) If 0, adaptative", type=float)
	parser.add_argument('--dropout_rate', default=0.5, help="Dropout rate (default: 0.5) 1.0 no dropout", type=float)
	parser.add_argument('--filter_std', default=0.1, help="standard deviation for the filters (default: 0.1)", type=float)
	parser.add_argument('--bias_ini', default=0.0, help="initialization of the bias (default: 0.0)", type=float)
	parser.add_argument('--repetitionPercentage', default=0.25, help="repetition percentage for the minibatch selection (default: 0.25)", type=float)
	parser.add_argument('--save_dir', default=None, help="checkpoint & summaries save dir name extension in tmp(default: None)")
	parser.add_argument('--gpu_num', default=0, help="CUDA visible device (default: 0)")
	parser.add_argument("--model_name", default="basic", help="basic / batchsel1 / batchsel2 / batchsel1prob / batchsel2prob (default: basic)")
	parser.add_argument("--augmentation_type", default="none", help="none / flip  (default: none)")
	parser.add_argument("--load_exist_model", default=False, help="Load pretrained model to exist model (default: false)", type=bool)
	parser.add_argument("--load_dir", default=None,help="Directory to load pretrained model (default: dataset_deep_method)")
	parser.add_argument("--dataset", default="mnist", help="Dataset to be trained; mnist / cifar10 / tinyImagenet / svhn (default: mnist)")
	parser.add_argument("--deep", default="vgg11b", help="Original deep VGG model, or not; vgg11b / vgg11 / vgg16 /vgg7 (default: vgg11b)")
	parser.add_argument("--rescale", default=None, help="Rescale the size of the images; 224  (default: None)", type=int)
	parser.add_argument("--finetune", default=False, help="Finetune VGG trained on ImageNet (vgg16 deep)", type=bool)
	parser.add_argument("--layers", default=3, help="Number of layers for the finetune for vgg16 (default: 3) 1 / 2 / 3 / 4 / 5", type=int)
	parser.add_argument("--subset", default=None, help="Number max of samples per class in the trainig set (default: None) ", type=int)

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
repetitionPercentage = FLAGS.repetitionPercentage

# Finetune
# noload_parameters = ['fc8_W', 'fc8_b', 'fc7_W', 'fc7_b', 'fc6_W', 'fc6_b', 'conv5_2_b', 'conv5_2_W', 'conv5_1_b', 'conv5_1_W', 'conv4_2_b', 'conv4_2_W', 'conv4_1_b', 'conv4_1_W', 'conv3_2_b', 'conv3_2_W', 'conv3_1_b', 'conv3_1_W', 'conv2_1_b', 'conv2_1_W', 'conv1_1_b', 'conv1_1_W']
# train_layers = ['fc8', 'fc7', 'fc6', 'conv5_2', 'conv5_1', 'conv4_2', 'conv4_1', 'conv3_2', 'conv3_1', 'conv2_1', 'conv1_1']
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
elif dataset == "cifar10":
	import cifar10

	cifar10.data_path = "data/CIFAR-10/"

	# cifar10.maybe_download_and_extract()	# Error en python 2

	class_names = cifar10.load_class_names()
	print(class_names)

	images_train, cls_train, labels_train = cifar10.load_training_data()
	images_test, cls_test, labels_test = cifar10.load_test_data()

	if FLAGS.subset != None:
		images_train, cls_train, labels_train = cifar10.load_training_data_subset(images_train, cls_train, FLAGS.subset)

	im_size = 32
	n_classes = 10
	im_channels = 3
	mean_values = [0.5623179,  0.5562303,  0.52719558]

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

	im_size = 32
	n_classes = 10
	im_channels = 3
	mean_values = [0.4376821,  0.4437697,  0.47280442]

	# Covert to one hot
	y_train = convert_labels_to_one_hot(y_train, n_classes)
	y_test = convert_labels_to_one_hot(y_test, n_classes)

#elif dataset == "lsun":

elif dataset=="tinyImagenet":
	train_file = 'data/TINYIMAGENET/train.txt'
	val_file = 'data/TINYIMAGENET/val.txt'
	classes_file = 'data/TINYIMAGENET/synset_words.txt'

	with open(classes_file) as f:
		lines = f.readlines()
		n_classes = len(lines)

	im_size = 224
	im_channels = 3
	mean_values = [0,  0,  0]

# batch = mnist.train.next_batch(batch_size)
# batch_img = batch[0].reshape((-1, im_size, im_size, im_channels))
# batch_lbl = batch[1]
#
# print(batch_img.shape, batch_lbl.shape)
#
# print(np.argmax(batch_lbl[0]))
# print(np.argmax(batch_lbl[1]))
# print(np.argmax(batch_lbl[2]))
# print(np.argmax(batch_lbl[3]))
#
#
# plt.figure()
# plt.imshow(batch_img[0, :, :, 0])
# plt.show()

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
	elif FLAGS.dataset == "tinyImagenet":
		train_generator = ImageDataGenerator(class_list=train_file, horizontal_flip=horizontal_flip, shuffle=True, scale_size=(im_size, im_size), nchannels=im_channels, nb_classes=n_classes, txt=True)
		val_generator = ImageDataGenerator(class_list=val_file, horizontal_flip=False, shuffle=False, scale_size=(im_size, im_size), nchannels=im_channels, nb_classes=n_classes, txt=True)
	elif FLAGS.dataset == "svhn":
		train_generator = ImageDataGenerator(x_train, y_train, horizontal_flip=horizontal_flip, shuffle=True, scale_size=(im_size, im_size), nchannels=im_channels,nb_classes=n_classes, inverse=True)
		val_generator = ImageDataGenerator(x_test, y_test, horizontal_flip=False, shuffle=False, scale_size=(im_size, im_size), nchannels=im_channels, nb_classes=n_classes, inverse=True)

	#train_generator.compute_mean()
	#[ 0.13070031] MNIST
	#[ 0.49139968  0.48215841  0.44653091] CIFAR10
	#[ 0.5623179 ,  0.5562303 ,  0.52719558] SVHN
	# tinyImagenet

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
	model = VGG_based_net(x, mean_values, keep_prob, n_classes, filter_std=filter_std, bias_ini=bias_ini, deep=FLAGS.deep)

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
		loss = tf.reduce_mean(loss_n)

	# Train op
	with tf.name_scope("train"):
		# Get gradients of all trainable variables
		gradients = tf.gradients(loss, var_list)
		gradients = list(zip(gradients, var_list))

		# Create optimizer and apply gradient descent to the trainable variables
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
		# model.load_initial_weights(sess)
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


			# model.save_npy(sess, npy_path="./vgg11-save.npy")

			np.save(filewriter_path + "/ValAcc.npy", np.array(test_acc_array))

		count = train_generator.getCount()

		np.save(filewriter_path + "/count.npy", np.array(count))


if __name__ == '__main__':
	train()

