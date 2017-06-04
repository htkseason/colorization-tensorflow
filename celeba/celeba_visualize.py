import tensorflow as tf
from tensorflow.python.framework import ops  
import matplotlib.pyplot as plt
import numpy as np
import celeba_input
import matplotlib
from PIL import Image


image = tf.image.rgb_to_grayscale(celeba_input.inputs(128)/255)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()

for i in range(100000):
    img = image.eval()
    print((img[0]/img[1]).shape)
    plt.imshow(img[0,:,:,0],cmap='gray')
    plt.show()
    print(i)