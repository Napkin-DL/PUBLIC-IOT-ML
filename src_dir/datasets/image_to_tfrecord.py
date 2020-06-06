# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Converts image data to TFRecords of TF-Example protos.

This module image data, uncompresses it, reads the files
that make up the image data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import glob
import cv2
import random

import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils


# The number of training files.
_NUM_TRAIN_FILES = 1

# The height and width of each image.
_IMAGE_SIZE = 128


def _add_to_tfrecord(filename, class_names_to_labels, tfrecord_writer, offset=0):
    """Loads data from the image_path and writes files to a TFRecord.

    Args:
      filename: The filename of the cifar10 pickle file.
      tfrecord_writer: The TFRecord writer to use for writing.
      offset: An offset into the absolute number of images previously written.

    Returns:
      The new offset.
    """
    # with tf.gfile.Open(filename, 'rb') as f:
    #     if sys.version_info < (3,):
    #         data = cPickle.load(f)
    #     else:
    #         data = cPickle.load(f, encoding='bytes')

    # images = data[b'data']
    # num_images = images.shape[0]

    # images = images.reshape((num_images, 3, 32, 32))
    # labels = data[b'labels']

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_jpeg(image_placeholder)

        with tf.Session('') as sess:
            num_images = len(filename)
            for j, images in enumerate(filename):
                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                    filename, offset + j + 1, offset + num_images))
                sys.stdout.flush()
                label = class_names_to_labels[images.split('/')[-2]]
                images = cv2.imread(images)
                image = cv2.resize(images, dsize=(
                    _IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                # image = np.squeeze(images).transpose((1, 2, 0))
                

                jpg_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})

                example = dataset_utils.image_to_tfexample(
                    jpg_string, b'jpg', _IMAGE_SIZE, _IMAGE_SIZE, label)
                tfrecord_writer.write(example.SerializeToString())

    return offset + num_images


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/captureddata_%s.tfrecord' % (dataset_dir, split_name)


# def _download_and_uncompress_dataset(dataset_dir):
#     """Downloads cifar10 and uncompresses it locally.

#     Args:
#       dataset_dir: The directory where the temporary files are stored.
#     """
#     filename = _DATA_URL.split('/')[-1]
#     filepath = os.path.join(dataset_dir, filename)

#     if not os.path.exists(filepath):
#         def _progress(count, block_size, total_size):
#             sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#                 filename, float(count * block_size) / float(total_size) * 100.0))
#             sys.stdout.flush()
#         filepath, _ = urllib.request.urlretrieve(
#             _DATA_URL, filepath, _progress)
#         print()
#         statinfo = os.stat(filepath)
#         print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
#         tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
    tf.gfile.DeleteRecursively(tmp_dir)


def run(image_path, dataset_dir):
    """Runs the download and conversion operation.

    Args:
      image_path : the image directory (input)
      dataset_dir: The dataset directory where the dataset is stored. (output)
    """

    if not tf.gfile.Exists(image_path):
        print("Image Path doesn't exist.")
        
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    image_list = glob.glob(os.path.join(image_path, '*/*'))
    random.shuffle(image_list)
    total_cnt = len(image_list)
    test_cnt = int(total_cnt/5) if int(total_cnt/5) > 0 else 1
    train_cnt = total_cnt - test_cnt

    train_img_list = image_list[:train_cnt]
    test_img_list = image_list[train_cnt:]

    # Finally, write the labels file:
    class_name = ['background']
    for label_item in glob.glob(os.path.join(image_path, '*')):
        class_name.append(label_item.split('/')[-1])

    labels_to_class_names = dict(zip(range(len(class_name)), class_name))

    class_names_to_labels = {}
    for i in labels_to_class_names.keys():
        class_names_to_labels[labels_to_class_names[i]] = i
        
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)


    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'val')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        offset = 0
        # for i in range(_NUM_TRAIN_FILES):
        #     filename = os.path.join(dataset_dir,
        #                             'captured_img',
        #                             'data_batch_%d' % (i + 1))  # 1-indexed.
        offset = _add_to_tfrecord(train_img_list, class_names_to_labels, tfrecord_writer, offset)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        # filename = os.path.join(dataset_dir,
        #                         'captured_img',
        #                         'test_batch')
        _add_to_tfrecord(test_img_list, class_names_to_labels, tfrecord_writer)

#     _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the image dataset!')
    return train_cnt, test_cnt
