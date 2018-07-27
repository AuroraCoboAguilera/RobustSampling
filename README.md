# RobustVGG
Study of the variance-based estimation for the selection of the samples in the mini-batch in the convolutional neural network VGG

This repository contains all the code needed to train some dataset with the different robust approaches of the selection of samples in the mini-batch.
For more information, used the TFM paper with the results for the mnist dataset and the explanation of the approaches.

`train_VGGbased_net.py` is the main program and have different arguments to configure the parameters of the network and the dataset to train, even if you desire to apply a finetune process.

- --num_epochs: (default 150) Number of training epochs in the training
- --batch_size: (default 128) Number of samples that belongs to a mini-batch.
- --learning_rate: (default 0.001) Learning rate for the optimizer. If 0, it is used an adaptative lerning rate that starts with the value of 0.01 and decreases at certain numbers of epochs. More precisely, divides in log scale the total number of epochs in 3 stages, and then change to 0.001 after the first stage and to 0.0001 after the second stage.
- --dropout_rate: (default 0.5) Configuration of the regularization by dropout. In order to remove it, configure its value to 1.0.
- --filter_std: (default 0.1) Standard deviation for the normal initialization of the parameters of the filters in the network, both convolutional filters and the ones of the fully connected layers.
- --bias_ini: (default 0.0) Constant initialization for the bias of the filters in all the layers.
- --repetitionPercentage: (default 0.25) Rate of samples that are repeated in one of the robust methods.
- --save_dir: (default None) Extension to the base for the name directory to save the results. The base name is the concatenation of the dataset with the network with the method. For example mnist_vgg11b_basic would be the base name for the training of the mnist dataset with the vgg11b network and the baseline method (without robust approach).
- --gpu_num: (default 0) Number of the GPU to be used by the nvidia commands. It is as it was executed the python script after the command CUDA_VISIBLE_DEVICES = 0
- --model_name: It is the approach to be used in the selection of the samples in the mini-batch. There are 5 choices: basic / batchsel1 / batchsel2 / batchsel1prob / batchsel2prob, where the first one is the baseline, the method number 1 is the VR-M, the method 2 is the VR-E and the prob refers to the probabilistic approach.
- --augmentation_type_ (default 'none'): It has two options 'none' and 'flip'. With 'flip', the images suffer an horizontal flip with probability 0.5 when calling the next mini-batch in the training.
- --load_exist_model: (default False) True / False. If True, it is continued a previous training of the same network by providing the checkpoint files.
- --load_dir: (default None) Extension to the directory to load the parameters of the pre-trained model. The default extension is none and the base name is the concatenation of the dataset with the network with the method.
- --dataset: (default mnist) mnist / cifar10 / tinyImagenet / svhn. Name of the dataset to be trained. It is needed the data in a folder with the name of the dataset in uppercase inside the folder 'data'.
- --deep: (default 'vgg11b') vgg11b / vgg11 / vgg16 / vgg7 / vgg10. Selection of the version of the VGG network as defined in `VGGbased_net.py`
- --rescale: (default None) New size of the images if it is desired them to be resized.
- --finetune: (default False) True / False. Boolean to finetune the network vgg16 pretrained on ImageNet.
- --layers: (default 3) 1 / 2 / 3 / 4 / 5. Number of the last layers to be trained if it is chosen to do a finetuning.
- --subset: (defaults None) Maximum number of samples per class in the training set. Only implemented wit cifar10.
