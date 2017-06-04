import time
import tensorflow as tf
import numpy
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import convert_to_tensor


from celeba import celeba_input

from colorization.colorizor import Cnet

record_log = True
record_log_dir = './log/colornet_absloss/'


iters = int(-1)

is_training = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.0002, global_step, 500, 1, staircase=False)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

# ========================

image64 = celeba_input.inputs(64)  # 0~255
image64_gray = tf.image.rgb_to_grayscale(image64) / 127.5 - 1.0  # -1~1
image64 = image64 / 127.5 - 1.0  # -1~1



with tf.variable_scope('cnet'):
    cnet = Cnet(image64_gray, is_training)

with tf.variable_scope('loss'):
    weight_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    square_loss = tf.reduce_mean(tf.abs(cnet.output_image - image64))
    accuarcy = tf.reduce_mean(tf.abs(cnet.output_image - image64))
    summary_losses = [tf.summary.scalar('square_loss', square_loss)]
# ===================

# train_step = tf.train.AdamOptimizer(learning_rate).minimize(square_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnet"), global_step=global_step)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(square_loss, global_step=global_step)

# ===================
merged = tf.summary.merge([summary_losses, tf.summary.image('gray_image', image64_gray, 64), tf.summary.image('image_gt', image64, 64),
                           tf.summary.image('colored_image', cnet.output_image, 64)])


sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()



# ========restore===============
saver = tf.train.Saver()
# saver.restore(sess, tf.train.get_checkpoint_state('../log/dcgan_grayscale').model_checkpoint_path)
# tf.train.write_graph(sess.graph_def, "./log/", "graph.pb", as_text=True);

# builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
# builder.add_meta_graph_and_variables(sess, ["tf-muct"])
# builder.save()

# ============train=================


def train():
    if record_log:
        log_writer = tf.summary.FileWriter(record_log_dir, sess.graph)
    start_time = time.time()    
    
    while True:
        
        [ _ ] = sess.run([ train_step], feed_dict={ is_training: True})
    
        if global_step.eval() % 100 == 0 :
            print('step = %d, lr = %g, time = %g min' % (global_step.eval(), learning_rate.eval(), (time.time() - start_time) / 60.0))
            [ acc , summ] = sess.run([ accuarcy, merged], feed_dict={is_training: False})
            print(acc)
            if record_log:
                log_writer.add_summary(summ, global_step.eval())
            print("==================")
            
        if global_step.eval() % 500 == 0 :
            # [gene_imgs] = sess.run([ gnet.layer4.output], feed_dict={ is_training: False})
            # plt.imshow((gene_imgs[0] + 1.0) / 2.0)
            # plt.show()
            saver.save(sess, os.path.join(record_log_dir, 'model.ckpt'), global_step.eval())
    
    
    
    print('total time = ', time.time() - start_time, 's')    
    if record_log:
        log_writer.close();    
        pass
    
