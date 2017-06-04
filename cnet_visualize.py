import os
import tensorflow as tf
from celeba import celeba_input
import numpy
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from celeba_color import *


saver.restore(sess, tf.train.get_checkpoint_state(record_log_dir).model_checkpoint_path)
#====testing
[img_gray, img_gt, img_colored] = sess.run([image64_gray, image64, cnet.output_image], feed_dict={is_training : False})


for i in range(img_gray.shape[0]):
    result = numpy.zeros([64, 64 * 3, 3])
    result[:, 0:64, :] = tf.image.grayscale_to_rgb(img_gray[i]).eval()
    result[:, 64:64*2, :] = img_gt[i]
    result[:, 64*2:64*3, :] = img_colored[i]
    plt.imshow((result + 1.0) / 2.0)
    plt.show()
