#!/usr/bin/env python
import argparse
import sys
import os
import logging
import numpy as np

def isImageValid(imagePath):
    # im = Image.open(imagePath)
    # width, height = im.size
    # minSize = 100
    # if width <= minSize or height <= minSize:
        # print 'Too small, so do not include :' + imagePath
        # return False
    return True

def scanAllImages(folderPath):
    extensions = {'.jpeg','.jpg', '.png'}
    images = []
    for root, dirs, files in os.walk(folderPath):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                relatviePath = os.path.join(root, file)
                if isImageValid(relatviePath) == True:
                    images.append(os.path.abspath(relatviePath))
    return images

def labelImagesInDir(label, folderPath):
    if label is None or folderPath is None:
        return

    # scan all images's abspath
    images = scanAllImages(folderPath)
    
    size_of_train = np.floor(args.size_of_train*len(images))
    size_of_val = np.floor(args.size_of_val*len(images))
    size_of_test = np.floor(args.size_of_test*len(images))

    countOfTrainImg = 0
    countOfValImg = 0
    countOfTestImg = 0

    train_file = open(args.file_train, 'a')
    val_file = open(args.file_val, 'a')
    test_file = open(args.file_test, 'a')
    for imagePath in images:
        if countOfTrainImg < size_of_train:
            train_file.write(imagePath + ' ' + str(label) + '\n')
            countOfTrainImg += 1
        elif countOfTestImg < size_of_test:
            test_file.write(imagePath + ' ' + str(label) + '\n')
            countOfTestImg += 1
        elif countOfValImg < size_of_val:
            val_file.write(imagePath + ' ' + str(label) + '\n')
            countOfValImg += 1

    train_file.close()
    val_file.close()
    test_file.close()

    if countOfTrainImg > 0 and countOfTrainImg < size_of_train:
        logging.warning('label ' + str(label) + ' only has ' + str(countOfTrainImg) + 'images for training')
    if countOfTestImg > 0 and countOfTestImg < size_of_test:
        logging.warning('label ' + str(label) + ' only has ' + str(countOfTestImg) + 'images for testing')
    if countOfValImg > 0 and countOfValImg < size_of_val:
        logging.warning('label ' + str(label) + ' only has ' + str(countOfValImg) + 'images for validation')



def autolableAllDir(path='.'):
    subdirectories = os.listdir(path)
    subdirectories.sort(key = str.lower)
    if args.file_test in subdirectories:
        bRemove = input(args.file_test + ' has already existed, should remove it? Y or N:')
        if 'Y'.lower() == bRemove:
            os.remove(args.file_test)
            if os.path.exists(args.file_test):
                print('Rmoeve fail')

    if args.file_train in subdirectories:
        bRemove = input(args.file_train + 'has already existed, should remove it? Y or N:')
        if 'Y'.lower() == bRemove:
            os.remove(args.file_train)
            if os.path.exists(args.file_train):
                print('Rmoeve fail')

    label = 0
    label_file = open('synset_words.txt', 'w')
    for subDir in subdirectories:
        if subDir.startswith('.'):#not os.path.isdir(subDir) or subDir.startswith('.'):
            print(subDir)
            continue
        labelImagesInDir(label, path + '/' + subDir)
        label = label + 1
        label_file.write(subDir + '\n')

    label_file.close()
    logging.warning('Total label size ' + str(label))

if __name__ == '__main__':
    if os.path.exists('log'):
        os.remove('log')
    logging.basicConfig(filename='log',level=logging.DEBUG)
    p = argparse.ArgumentParser(description='Create a label to train.txt, val.txt, or test.txt')
    p.add_argument('--dir', '-d', help='Image label that will be saved to train.txt or test.txt')
    p.add_argument('--label', '-l', type=int, help='Image label that will be saved to train.txt or test.txt')
    p.add_argument('--size_of_train', type=float, default=0.8, help='put how many image to train.txt')
    p.add_argument('--size_of_val', type=float, default=0.2, help='put how many image to val.txt')
    p.add_argument('--size_of_test', type=float, default=0.0, help='put how many image to text.txt')
    p.add_argument('--file_train', default = 'train.txt', help='The filename for train. By default, train.txt')
    p.add_argument('--file_val', default = 'val.txt', help='The filename for validation. By default, val.txt')
    p.add_argument('--file_test', default = 'test.txt', help='The filename for test. By default, test.txt')
    p.add_argument('--verbose', '-v', action='store_true', help='Enable verbose log')
    args = p.parse_args()
    if args.label is None or args.dir is None:
        # bAutoLabelAllDir = input("Should label all dirs? Y or N :")
        # if 'Y'.lower() == bAutoLabelAllDir:
        print('Automatically label all image in each folder under the dir')
        if args.dir is not None:
            autolableAllDir(args.dir)
        else:
            autolableAllDir()
    else:
        labelImagesInDir(args.label, args.dir)

