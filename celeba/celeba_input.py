from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from PIL import Image

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 202600
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0

min_fraction_of_examples_in_queue = 0.2
data_dir = 'e:/CelebA/Img/img_align_celeba/'
tfrecord_dir = 'e:/CelebA/Img/img_tfrecord/'
packages = 20


def generate_tfrecord():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
   
    reader = tf.WholeFileReader()
    _, value = reader.read(tf.train.string_input_producer([os.path.join(data_dir, '%s' % img_name) for img_name in os.listdir(data_dir)  ]))
    image = tf.image.decode_jpeg(value)
    image = tf.image.crop_to_bounding_box(image, 30, 0, 178, 178)
    image = tf.cast(tf.image.resize_images(image, [64, 64], method=tf.image.ResizeMethod.AREA), tf.uint8)
    tf.train.start_queue_runners()
    for file_index in range(packages):
        writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, 'celeba_image_%d.tfrecords' % file_index))
        for i in range(int(file_index * (202600 / packages)), int((file_index + 1) * (202600 / packages))):
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.eval().tobytes()]))
                }))
            # plt.imshow(image.eval())
            # plt.show()
            writer.write(example.SerializeToString())  # 序列化为字符串
            print(i)
        writer.close()

# generate_tfrecord()


def inputs(batch_size):  

    filename_queue = tf.train.string_input_producer([os.path.join(tfrecord_dir, 'celeba_image_%d.tfrecords' % i) for i in range(0, packages)])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_example([serialized_example],
                                       features={
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.cast(tf.reshape(image, [64, 64, 3]), tf.float32)
    return tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=16,
            capacity=int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue + 3 * 128),
            min_after_dequeue=int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue))


