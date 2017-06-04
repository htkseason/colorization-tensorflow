
import tensorflow as tf
import numpy
import math
from tfobjs import *

class Cnet:

    def __init__(self, gray_image_tensor, is_training):
        
        with tf.variable_scope('in_layer0'):
            self.ilayer0 = ConvObj()
            self.ilayer0.set_input(gray_image_tensor)
            self.ilayer0.batch_norm(self.ilayer0.conv2d([5, 5], 32, [1, 1, 1, 1]), is_training=is_training)
            self.ilayer0.set_output(tf.nn.relu(self.ilayer0.bn))
        # ===================64-64
        
        with tf.variable_scope('in_layer1'):
            self.ilayer1 = ConvObj()
            self.ilayer1.set_input(self.ilayer0.output)
            self.ilayer1.batch_norm(self.ilayer1.conv2d([5, 5], 64, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer1.set_output(tf.nn.relu(self.ilayer1.bn))
        # ===================64->32

        with tf.variable_scope('in_layer2'):
            self.ilayer2 = ConvObj()
            self.ilayer2.set_input(self.ilayer1.output)
            self.ilayer2.batch_norm(self.ilayer2.conv2d([5, 5], 128, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer2.set_output(tf.nn.relu(self.ilayer2.bn))
        # ===================32->16

        with tf.variable_scope('in_layer3'):
            self.ilayer3 = ConvObj()
            self.ilayer3.set_input(self.ilayer2.output)
            self.ilayer3.batch_norm(self.ilayer3.conv2d([5, 5], 256, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer3.set_output(tf.nn.relu(self.ilayer3.bn))
        # ===================16->8

        with tf.variable_scope('in_layer4'):
            self.ilayer4 = ConvObj()
            self.ilayer4.set_input(self.ilayer3.output)
            self.ilayer4.batch_norm(self.ilayer4.conv2d([5, 5], 512, [1, 2, 2, 1]), is_training=is_training)
            self.ilayer4.set_output(tf.nn.relu(self.ilayer4.bn))
        # ===================8->4
        
        with tf.variable_scope('out_layer1'):
            self.olayer1 = ConvObj()
            self.olayer1.set_input(self.ilayer4.output)
            self.olayer1.batch_norm(self.olayer1.deconv2d([5, 5], [8, 8, 256], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer1.set_output(tf.nn.relu(self.olayer1.bn))
        # ===================4->8

        with tf.variable_scope('out_layer2'):
            self.olayer2 = ConvObj()
            self.olayer2.set_input(self.olayer1.output)
            self.olayer2.batch_norm(self.olayer2.deconv2d([5, 5], [16, 16, 128], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer2.set_output(tf.nn.relu(self.olayer2.bn))
        # ===================8->16
    
        with tf.variable_scope('out_layer3'):
            self.olayer3 = ConvObj()
            self.olayer3.set_input(self.olayer2.output)
            self.olayer3.batch_norm(self.olayer3.deconv2d([5, 5], [32, 32, 64], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer3.set_output(tf.nn.relu(self.olayer3.bn))
        # ===================16->32
    
        with tf.variable_scope('out_layer4'):
            self.olayer4 = ConvObj()
            self.olayer4.set_input(self.olayer3.output)
            self.olayer4.batch_norm(self.olayer4.deconv2d([5, 5], [64, 64, 32], strides=[1, 2, 2, 1]), is_training=is_training)
            self.olayer4.set_output(tf.nn.relu(self.olayer4.bn))
        # ===================32->64
        
        with tf.variable_scope('out_layer5'):
            self.olayer5 = ConvObj()
            self.olayer5.set_input(tf.concat([self.olayer4.output, self.ilayer0.output] , axis=3))
            self.olayer5.batch_norm(self.olayer5.deconv2d([5, 5], [64, 64, 3], strides=[1, 1, 1, 1]), is_training=is_training)
            self.olayer5.set_output(tf.nn.tanh(self.olayer5.logit))
        # ===================64(x)->64(3)
        
        self.output_image = self.olayer5.output 
