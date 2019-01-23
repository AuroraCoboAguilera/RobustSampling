##################################################################################################################################
#
# Data generator for images read directly from a python library. You can read from txt with class_list,
# o directly process the images with txt=False and using X,y
#
#
# This code is highly influenced by the implementation
# https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d
#
# I add the particular functions to generate the next mini-batches according to the VR-M and VR-E.
#
#
#
##################################################################################################################################


import numpy as np
import cv2
from scipy.misc import imread, imresize



class ImageDataGenerator:
	def __init__(self, X=None, y=None, class_list=None, horizontal_flip=False, shuffle=False, scale_size=(224, 224), nchannels=3, nb_classes=2, txt=False, inverse=False):

		# Init params
		self.horizontal_flip = horizontal_flip
		self.n_classes = nb_classes
		self.shuffle = shuffle
		self.scale_size = scale_size
		self.pointer = 0
		self.nchannels = nchannels
		self.txt = txt
		self.inverse = inverse

		self.read_class_list(X, y, class_list)

		if self.shuffle:
			self.shuffle_data()

	def read_class_list(self, X, y, class_list):
		"""
		Scan the image file and get the image paths and labels
		"""
		self.images = []
		self.labels = []
		self.indexs = []
		self.count = []
		if self.txt:
			with open(class_list) as f:
				lines = f.readlines()
				i = 0
				for l in lines:
					items = l.split()
					self.images.append(items[0])
					self.labels.append(int(items[1]))
					self.count.append(0)
					self.indexs.append(i)
					i = i + 1
		else:
			if self.inverse == False:
				for i in range(len(y)):
					self.images.append(X[i, :, :, :])
					self.labels.append(y[i, :])
					self.count.append(0)
					self.indexs.append(i)
			else:
				for i in range(len(y)):
					self.images.append(X[:, :, :, i])
					self.labels.append(y[i, :])
					self.count.append(0)
					self.indexs.append(i)
            
		#store total number of data
		self.data_size = len(self.labels)

		self.images_select = []
		self.labels_select = []
		self.indexs_select = []

	def shuffle_data(self):
		"""
		Random shuffle the images and labels
		"""
		images = self.images[:]#self.images.copy()
		labels = self.labels[:]#self.labels.copy()
		indexs = self.indexs[:]
		self.images = []
		self.labels = []
		self.indexs = []
        
		#create list of permutated index and shuffle data accoding to list
		idx = np.random.permutation(len(labels))
		for i in idx:
			self.images.append(images[i])
			self.labels.append(labels[i])
			self.indexs.append(indexs[i])

	def reset_pointer(self):
		"""
		reset pointer to begin of the list
		"""
		self.pointer = 0

		if self.shuffle:
			self.shuffle_data()
        

	def next_batch(self, batch_size):
		"""
		This function gets the next n ( = batch_size) images from the path list
		and labels and loads the images into them into memory
		"""
		# Get next batch of image (path) and labels
		originalImgs = self.images[self.pointer:self.pointer + batch_size]
		labels = self.labels[self.pointer:self.pointer + batch_size]
		indexs = self.indexs[self.pointer:self.pointer + batch_size]

		#update pointer
		self.pointer += batch_size

		# Read images
		images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], self.nchannels])
		for i in range(len(originalImgs)):
			if self.txt:
				img = imread(originalImgs[i], mode='RGB')
			else:
				img = originalImgs[i]

			#flip image at random if flag is selected
			if self.horizontal_flip and np.random.random() < 0.5:
				img = cv2.flip(img, 1)

			#rescale image
			if self.txt:
				img = imresize(img, (self.scale_size[0], self.scale_size[1]))
			else:
				img = np.reshape(img, (self.scale_size[0], self.scale_size[1], self.nchannels))
			img = img.astype(np.float32)	# it changes the visualization!!!!

			#subtract mean
			#img -= self.mean

			images[i] = img

		# Expand labels to one hot encoding
		if self.txt:
			one_hot_labels = np.zeros((batch_size, self.n_classes))
			for i in range(len(labels)):
				one_hot_labels[i][labels[i]] = 1
		else:
			one_hot_labels = np.array(labels).astype(np.float32)

		# add the count
		for i in range(len(indexs)):
			self.count[indexs[i]] = self.count[indexs[i]] + 1

		#return array of images and labels
		return images, one_hot_labels


	# compute the mean
	def compute_mean(self):
		mean = np.zeros(self.nchannels)
		for i in range(len(self.images)):
			if self.txt:
				mean += np.mean(imread(self.images[i]), axis=(0, 1))
			else:
				mean += np.mean(self.images[i], axis=(0, 1))
		mean /= len(self.images)

		return mean


	def next_batch_robust(self, batch_size, loss_previous, repetitionPercentage, first=True, prob=False):
		"""
		This function gets the next n ( = batch_size) images from the path list
		and labels and loads the images into them into memory
		loss_previous: The loss obtained for the previous minibatch
		repetitionPercentage: The percentage of samples of the previous dataset to be repeated, [0, 1]
		It is used by the VR-M.
		"""

		# Get next batch of image (path) and labels
		originalImgs = self.images[self.pointer:self.pointer + batch_size]
		labels = self.labels[self.pointer:self.pointer + batch_size]
		indexs = self.indexs[self.pointer:self.pointer + batch_size]

		# In the first iteration of each epoch, we do not have to replace any samples.
		if first:
			self.images_select = self.images[:batch_size]
			self.labels_select = self.labels[:batch_size]
			self.indexs_select = self.indexs[:batch_size]
		else:
			# Number of samples to be repeated in this next batch
			n_repetitionSamples = int(np.floor(repetitionPercentage*batch_size))

			# Get the n samples worst classified
			index_sorted = np.argsort(loss_previous)[::-1][:n_repetitionSamples]

			# Probabilistic method to avoid outlayers, take randomly the half of the worse trained samples
			if prob:
				# Compute the new number of repeated samples
				n_repetitionSamples = int(np.floor(n_repetitionSamples/2))
				# Randomly shuffle the array with the index of the samples to repeat, in order to later take the first random ones instead of the worst classified
				np.random.shuffle(index_sorted)

			# Get n random indexs to replace the current samples with the worst classified ones (or with the sampled worst classified)
			index_replace = np.arange(batch_size)
			np.random.shuffle(index_replace)	# The original samples to be replaced are selected randomly
			index_replace = index_replace[:n_repetitionSamples]

			# Replace samples: the images, the labels and the indentification of the sample to take the count of times that has be repeated in the whole training
			for i, ind_r in enumerate(index_replace):
				originalImgs[ind_r] = self.images_select[index_sorted[i]]
				labels[ind_r] = self.labels_select[index_sorted[i]]
				indexs[ind_r] = self.indexs_select[index_sorted[i]]

			self.images_select = originalImgs[:]
			self.labels_select = labels[:]
			self.indexs_select = indexs[:]

		# update pointer
		self.pointer += batch_size

		# Read images
		images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], self.nchannels])
		for i in range(len(originalImgs)):
			if self.txt:
				img = imread(originalImgs[i], mode='RGB')
			else:
				img = originalImgs[i]

			#flip image at random if flag is selected
			if self.horizontal_flip and np.random.random() < 0.5:
				img = cv2.flip(img, 1)

			#rescale image
			if self.txt:
				img = imresize(img, (self.scale_size[0], self.scale_size[1]))
			else:
				img = np.reshape(img, (self.scale_size[0], self.scale_size[1], self.nchannels))
			img = img.astype(np.float32)  # it changes the visualization!!!!

			#subtract mean
			#img -= self.mean

			images[i] = img

		# Expand labels to one hot encoding
		if self.txt:
			one_hot_labels = np.zeros((batch_size, self.n_classes))
			for i in range(len(labels)):
				one_hot_labels[i][labels[i]] = 1
		else:
			one_hot_labels = np.array(labels).astype(np.float32)

		# add the count
		for i in range(len(indexs)):
			self.count[indexs[i]] = self.count[indexs[i]] + 1

		# return array of images and labels
		return images, one_hot_labels



	def replace_data(self, losses, repetitionPercentage, prob=False):
		"""
		Random replace the images and labels. Used in the VR-E method to replace images of a epoch.
		"""

		# Compute the number of samples to the repeated
		n_repetitionSamples = int(np.floor(repetitionPercentage * len(losses)))

		# Get the n samples worst classified
		index_sorted = np.argsort(losses)[::-1][:n_repetitionSamples]

		# Probabilistic method to avoid outlayers, take randomly the half of the worse trained samples
		if prob:
			n_repetitionSamples = int(np.floor(n_repetitionSamples / 2))
			np.random.shuffle(index_sorted)

		# Get n random indexs to replace the original samples with the worst classified ones
		idx_replace = np.random.permutation(len(self.labels))[:n_repetitionSamples]
		
		self.images_select = []
		self.labels_select = []
		self.indexs_select = []

		i = 0
		# Replace the samples
		for l in np.arange(len(self.labels)):
			if l in idx_replace:
				self.images_select.append(self.images[index_sorted[i]])
				self.labels_select.append(self.labels[index_sorted[i]])
				self.indexs_select.append(self.indexs[index_sorted[i]])
				i = i + 1
			else:
				self.images_select.append(self.images[l])
				self.labels_select.append(self.labels[l])
				self.indexs_select.append(self.indexs[l])


	def next_batch_robust_2(self, batch_size, first=True):
		"""
		This function gets the next n ( = batch_size) images from the path list
		and labels and loads the images into them into memory
		It is used by the VR-E with the images previously replaced by the replace_data() method
		"""
		# Get next batch of image (path) and labels
		if first:
			originalImgs = self.images[self.pointer:self.pointer + batch_size]
			labels = self.labels[self.pointer:self.pointer + batch_size]
			indexs = self.indexs[self.pointer:self.pointer + batch_size]
		else:
			originalImgs = self.images_select[self.pointer:self.pointer + batch_size]
			labels = self.labels_select[self.pointer:self.pointer + batch_size]
			indexs = self.indexs_select[self.pointer:self.pointer + batch_size]

		# update pointer
		self.pointer += batch_size

		# Read images
		images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], self.nchannels])
		for i in range(len(originalImgs)):
			if self.txt:
				img = imread(originalImgs[i], mode='RGB')
			else:
				img = originalImgs[i]

			# flip image at random if flag is selected
			if self.horizontal_flip and np.random.random() < 0.5:
				img = cv2.flip(img, 1)

			# rescale image
			if self.txt:
				img = imresize(img, (self.scale_size[0], self.scale_size[1]))
			else:
				img = np.reshape(img, (self.scale_size[0], self.scale_size[1], self.nchannels))
			img = img.astype(np.float32)  # it changes the visualization!!!!

			# subtract mean
			# img -= self.mean

			images[i] = img

		# Expand labels to one hot encoding
		if self.txt:
			one_hot_labels = np.zeros((batch_size, self.n_classes))
			for i in range(len(labels)):
				one_hot_labels[i][labels[i]] = 1
		else:
			one_hot_labels = np.array(labels).astype(np.float32)

		# add the count
		for i in range(len(indexs)):
			self.count[indexs[i]] = self.count[indexs[i]] + 1

		# return array of images and labels
		return images, one_hot_labels


	def getCount(self):
		return self.count



def convert_labels_to_one_hot(labels, num_labels):
	'''Function to convert labels to one hot format.'''
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return labels
