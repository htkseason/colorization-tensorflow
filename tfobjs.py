import tensorflow as tf
import numpy

def leaky_relu(x, leak, name=None):
    return tf.maximum(x, x * leak, name=name)

class BnObj():
    def __init__(self, shape, is_conv=True):
        self.is_conv = is_conv
        self.bn_beta = tf.get_variable('bn_beta', shape[-1], dtype=tf.float32, initializer=tf.zeros_initializer(tf.float32))
        self.bn_scale = tf.get_variable('bn_scale', shape[-1], dtype=tf.float32, initializer=tf.ones_initializer(tf.float32))
        self.bn_pop_mean = tf.get_variable('bn_pop_mean', shape[-1], dtype=tf.float32, trainable=False, initializer=tf.zeros_initializer(tf.float32))
        self.bn_pop_var = tf.get_variable('bn_pop_var', shape[-1], dtype=tf.float32, trainable=False, initializer=tf.ones_initializer(tf.float32))
    
    def batch_norm(self, inputs, is_training, decay=0.9):
        if is_training:
            if self.is_conv:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            else:
                batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(self.bn_pop_mean, self.bn_pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(self.bn_pop_var, self.bn_pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.bn_beta, self.bn_scale, 1e-5, name='bn_train')
        else:
            return tf.nn.batch_normalization(inputs, self.bn_pop_mean, self.bn_pop_var, self.bn_beta, self.bn_scale, 1e-5, name='bn_eval')


class ConvObj(object):
    def set_input(self, tensor):
        self.input = tensor
        self.input_size = tensor.shape.as_list()[1:]
    def set_output(self, tensor):
        self.output = tensor
        self.output_size = tensor.shape.as_list()[1:] 

    def conv2d(self, k_shape, layer_size, strides=[1, 1, 1, 1] , l2=0.0, padding='SAME'):
        self.weight = tf.get_variable('weight', k_shape + [self.input_size[-1] , layer_size], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(scale=l2))
        self.bias = tf.get_variable('bias', [layer_size], initializer=tf.constant_initializer(0.0))
        self.logit = tf.nn.bias_add(tf.nn.conv2d(self.input, self.weight, strides=strides, padding=padding), self.bias, name='logit')
        self.summary = [tf.summary.histogram('weight', self.weight), tf.summary.histogram('bias', self.bias),
                        tf.summary.histogram('logit', self.logit)]
        return self.logit
    
    def deconv2d(self, k_shape, output_shape, strides=[1, 2, 2, 1], l2=0.0, padding='SAME'):
        self.output_shape = output_shape
        self.weight = tf.get_variable('weight', k_shape + [output_shape[-1], self.input_size[-1]], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(scale=l2))
        self.bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        self.logit = tf.nn.bias_add(tf.nn.conv2d_transpose(self.input, self.weight, output_shape=[tf.shape(self.input)[0]] + output_shape,
                                     strides=strides, padding=padding), self.bias, name='logit')
        self.logit = tf.reshape(self.logit, [-1] + output_shape)
        self.summary = [tf.summary.histogram('weight', self.weight), tf.summary.histogram('bias', self.bias),
                        tf.summary.histogram('logit', self.logit)]
        return self.logit
    
        
    def batch_norm(self, tensor, is_training):
        self.bn_obj = BnObj(tensor.shape.as_list()[1:], is_conv=True)
        self.bn = tf.cond(is_training, lambda: self.bn_obj.batch_norm(tensor, True), lambda: self.bn_obj.batch_norm(tensor, False)) 
        return self.bn
    
    def lrn(self, tensor, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75):
        self.lrn = tf.nn.lrn(tensor, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name='lrn')
        return self.lrn
    

class FcObj(object):
    def set_input(self, tensor):
        self.input_size = numpy.cumprod(tensor.shape.as_list()[1:])[-1]
        self.input = tf.reshape(tensor, [-1, self.input_size], name='input')
    def set_output(self, tensor):
        self.output_size = numpy.cumprod(tensor.shape.as_list()[1:])[-1]
        self.output = tf.reshape(tensor, [-1, self.output_size], name='output')
        
    def fc(self, output_size, l2=0.0, is_training=None, keep_prob=1.0):
        self.weight = tf.get_variable('weight', [self.input_size, output_size], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True), regularizer=tf.contrib.layers.l2_regularizer(scale=l2))
        self.bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.0))
        self.logit = tf.nn.bias_add(tf.matmul(self.input, self.weight), self.bias, name='logit')
        if is_training is not None:
            self.logit = tf.cond(is_training, lambda : tf.nn.dropout(self.logit, keep_prob, name='logit_dropout') , lambda : self.logit)
        
        self.summary = [tf.summary.histogram('weight', self.weight), tf.summary.histogram('bias', self.bias),
                        tf.summary.histogram('logit', self.logit)]
        return self.logit
    
    def batch_norm(self, tensor, is_training):
        self.bn_obj = BnObj(tensor.shape.as_list()[1:], is_conv=False)
        self.bn = tf.cond(is_training, lambda: self.bn_obj.batch_norm(tensor, True), lambda: self.bn_obj.batch_norm(tensor, False)) 
        return self.bn
    
    def lrn(self, tensor, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75):
        self.lrn = tf.nn.lrn(tensor, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name='lrn')
        return self.lrn


