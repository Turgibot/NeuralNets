# Copyright 2018 Guy Tordjman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
########################################################################################################################


'''
Create a tfrecord from given data
As for today tfrecord is the recommended format for tensor flow
It is a binary file format - a serialized tf.tarin.Example protobuf object
better use of disk cache - faster to move around - can handle data of different types (image + label in one object)

good blogpost :  http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

Feature: an image
Label: a number

basic steps for a single image:

# Step 1: create a writer to write tfrecord to that file
writer = tf.python_io.TFRecordWriter(out_file)
# Step 2: get serialized shape and values of the image
shape, binary_image = get_image_binary(image_file)
# Step 3: create a tf.train.Features object
features = tf.train.Features(feature={'label': _int64_feature(label),
'shape': _bytes_feature(shape),
'image': _bytes_feature(binary_image)})
# Step 4: create a sample containing of features defined above
sample = tf.train.Example(features=features)
# Step 5: write the sample to the tfrecord file
writer.write(sample.SerializeToString())
writer.close()


steps specific for Kitti dataset
Step 1: Split
'''


import os
import sys
import re
import tensorflow as tf
import cv2
import numpy as np
import utils

# TFRecords convertion parameters.
RANDOM_SEED = 2018
SAMPLES_PER_FILES = 200
CWD = os.getcwd()
CODE_DIR = os.path.abspath(os.path.join(CWD, os.pardir))
ROOT_DIR = os.path.abspath(os.path.join(CODE_DIR, os.pardir))
PATH_TFRECORDS = os.path.join(CODE_DIR, 'TFRECORDS')
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
KITTY_DIR = os.path.join(DATA_DIR, 'Kitti')
PATH_IMAGES = os.path.join(KITTY_DIR, 'Images')
PATH_LABELS = os.path.join(KITTY_DIR, 'Labels')

CLASSES = {
    'Pedestrian': 0,
    'Cyclist': 1,
    'Car': 2,
}


def start():
    # create a folder for the tfrecords if doesn't exist'''
    utils.create_dir(PATH_TFRECORDS)
    train_labels, test_labels = utils.random_split_kitti(PATH_LABELS, 0.8, CLASSES, RANDOM_SEED)

    # Step 1: create a writer to write tfrecord to that file



start()