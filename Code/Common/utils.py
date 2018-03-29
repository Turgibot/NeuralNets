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
# ==============================================================================
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import sys
import cv2
# to create a tf.Feature object a conversion to tf.Train>Feature is required according to the correct dtype.
# first check if input value is an instance of tuple or list - if a scalar insert it to a list
# then return tf.train.Feature(<type>_list=tf.train.<type>List(value=values))
# continue by inserting the features into Example proto.


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


# image to basic tf.train.Example instance
def image_to_basic_example(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }, name='features'))

# image to KITTI dataset tf.train.Example instance
def image_to_kitti_example(image_data, shape, bb_list, label_list):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': image_data,
        'image/shape': shape,
        'image/boxes': bb_list,
        'image/labels': label_list
    }, name='features'))


def get_image_binary(filename):

    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape, image.tobytes()  # convert image to raw data bytes in the array.


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def create_file(path):
    try:
        open(path, 'a').close()
    except OSError:
        sys.stdout.write('creating a file error in '+path)


'''prepare kitti data with cross validation'''


def random_split_kitti(label_src, test_size=0.75, classes={}, seed=0):

    label_files = [f for f in os.listdir(label_src) if os.path.isfile(os.path.join(label_src, f))]
    processed_labels = {}
    files_count = len(label_files)
    random.seed(seed)
    random.shuffle(label_files)
    for label in label_files:
        name = label.split('.')[0]
        processed_labels[name] = []
        with open(os.path.join(label_src, label), 'r', encoding='utf-8')as l:
            lines = l.readlines()
            for line in lines:
                line_data = []
                info = line.split(' ')
                class_info = info[0]
                if class_info in classes:
                    line_data.append(classes[class_info])
                    line_data.append(float(info[4]))
                    line_data.append(float(info[5]))
                    line_data.append(float(info[6]))
                    line_data.append(float(info[7]))
                    processed_labels[name].append(line_data)

    training_count = int(test_size*files_count)
    training_labels = dict(list(processed_labels.items())[:training_count])
    testing_labels = dict(list(processed_labels.items())[training_count:])
    return training_labels, testing_labels


def append_to_tfrecord(image, label, writer):

    class_list = []
    bbox_x1_list = []
    bbox_y1_list = []
    bbox_x2_list = []
    bbox_y2_list = []

    for obj in label:
        class_num = obj[0]
        class_list.append(class_num)

        # Bounding Box
        bbox_x1 = float(obj[1])
        bbox_y1 = float(obj[2])
        bbox_x2 = float(obj[3])
        bbox_y2 = float(obj[4])
        bbox_x1_list.append(bbox_x1)
        bbox_y1_list.append(bbox_y1)
        bbox_x2_list.append(bbox_x2)
        bbox_y2_list.append(bbox_y2)



    shape, image_data = get_image_binary(image)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/object/bbox/xmin': float_feature(bbox_x1_list),
        'image/object/bbox/xmax': float_feature(bbox_x2_list),
        'image/object/bbox/ymin': float_feature(bbox_y1_list),
        'image/object/bbox/ymax': float_feature(bbox_y2_list),
        'image/object/bbox/label': int64_feature(class_list)
    }))

    writer.write(example.SerializeToString())
