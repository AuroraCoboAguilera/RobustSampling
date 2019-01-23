# Robust Sampling in Deep Learning
Study of the variance-based estimation for the selection of the samples in the mini-batch in convolutional neural networks.

This repository contains all the code needed to train some datasets and neural networks with the different robust approaches of the selection of samples in the mini-batch, VR-M and VR-E.
For more information, make use of the paper.

`train_VGGbased_net.py` is the main program and have different arguments to configure the parameters of the network and the dataset to train, even if you desire to apply a finetune process.

| **Argument** | **Meaning** |
|:-------------|:------------|
| --num_epochs (default 150) | Number of training epochs in the training. |
| --batch_size (default 128) | Number of samples that belongs to a mini-batch. |
| --learning_rate (default 0.001) | Learning rate for the optimizer. If 0, it is used an adaptive learning rate that starts with the value of 0.01 and decreases at certain numbers of epochs. More precisely, divides in log scale the total number of epochs in 3 stages, and then change to 0.001 after the first stage and to 0.0001 after the second stage. |
| --use_lr_schedule (default False)| Use learning rate schedule. It consists on dividing the learning rate by 10 every of the three epochs from the lr_schecule argument. |
| --lr_schedule (default [200, 250, 300]) | Learning rate schedule for the adaptive scheme. They are the epochs where the learning rate decreases. |
| --weight_decay (default 0.001) | Weight decay that can be used in ALL-CONV architecture. |
| --dropout_rate (default 0.5) | Configuration of the regularization by dropout. In order to remove it, choose its value to 1.0 |
| --filter_std (default 0.1) | Standard deviation for the normal initialization of the parameters of the filters in the network, both convolutional filters and the ones of the fully connected layers. |
| --bias_ini (default 0.0) | Constant initialization for the biases of the filters in all the layers. |
| --repetitionPercentage (default 0.25) | Rate of samples that are repeated in one of the robust methods. |
| --save_dir  (default None) | Extension to the base for the name directory to save the results. The base name is the concatenation of the dataset with the network with the method. For example mnist_vgg11b_basic would be the base name for the training of the MNIST dataset with the VGG11b network and the baseline method (without robust approach). |
| --gpu_num  (default 0) | Number of the GPU to be used by the NVIDIA commands. It is as it was executed the python script after the command configuration CUDA_VISIBLE_DEVICES = 0. |
| --model_name | It is the approach to be used in the selection of the samples in the mini-batch. There are 5 choices: basic / batchsel1 / batchsel2 / batchsel1prob / batchsel2prob, where the first one is the baseline, the method number 1 is the VR-M, the method 2 is the VR-E and the prob refers to the probabilistic approach. |
| --augmentation_type_ (default 'none') | It has two options 'none' and 'flip'. With 'flip', the images suffer an horizontal flip with probability 0.5 when calling the next mini-batch in the training. |
| --load_exist_model (default False) | True / False. If True, it is continued a previous training of the same network by providing the checkpoint files. |
| --load_dir (default None) | Extension to the directory to load the parameters of the pre-trained model. The default extension is none and the base name is the concatenation of the dataset with the network with the method. |
| --dataset (default mnist) | mnist / cifar10 / svhn. Name of the dataset to be trained. It is needed the data in a folder with the name of the dataset in uppercase inside the folder 'data'. |
| --deep (default 'vgg11b') | vgg11b / all-cnn / vgg16 / vgg7 / vgg10. Selection of the neural network architecture as defined in `VR_net.py`. |
| --rescale (default None) | New size of the images if it is desired them to be resized. |
| --preprocess (default False)| whitened and contrast normalized. Implemented for CIFAR-10. |
| --finetune (default False) | True / False. Boolean to finetune the network vgg16 pretrained on ImageNet (only with vgg16 deep). |
| --layers (default 3) | 1 / 2 / 3 / 4 / 5. Number of the last layers to be trained if it is chosen to do a finetuning. |
| --subset (default None) | Maximum number of samples per class in the training set. Only implemented with cifar10. |
| --train_samples (default: None, which is 55000 train samples and 10000 test samples in MNIST for example) | Number of training samples. |
| --regularizer (default: True)| Use the regularizer of the ALL-CNN architecture. Only implemented with all-cnn deep. |


An example for the utilization of the code for the different scenarios in the paper would be:
- python train_VR_net.py --num_epochs 500 --batch_size 128 --learning_rate 0.01 --use_lr_schedule True --preprocess True --dataset cifar10 --deep all-cnn --dropout_rate 0.5 --filter_std 0.05 --train_samples 30000 --model_name batchsel1 --repetition_percentage 0.1 --save_dir _10_pre_epochs500_batch128_dropout05_lr01_std05_train30000 --gpu_num 2

- python train_VGGbased_net.py --num_epochs 500 --batch_size 128 --dropout_rate 0.5 --dataset svhn --model_name batchsel2 --repetition_percentage 0.2 --save_dir _20_epochs500_batch128_dropout05  --gpu_num 0

- python train_VGGbased_net.py --num_epochs 200 --batch_size 64 --dropout_rate 0.5 --dataset mnist --model_name batchsel1prob --repetition_percentage 0.2 --train_samples 40000 --save_dir _20_epochs500_batch128_dropout05_train40000  --gpu_num 0
