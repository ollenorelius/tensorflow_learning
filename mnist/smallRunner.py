#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def showData(image):
    image = np.reshape(image,(28,28))
    plt.imshow(image)
    plt.show()
    return None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([3,3,1,16])
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3,3,16,16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*16])

W_fc1 = weight_variable([7*7*16, 128])
b_fc1 = bias_variable([128])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
out = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
saver.restore(sess, './networks/smallModel.cpt')

#batch = mnist.train.next_batch(200)

img = misc.imread('2.png')
img = 1-(img[:,:,1]/255.0)
img = np.reshape(img, [-1, 784])

sess.run(tf.global_variables_initializer())
import time
print("Started")
startTime = time.time()

for i in range(20):
    sess.run(tf.argmax(out,1), feed_dict={x:np.repeat(img,1000,axis=0), keep_prob:1.0})
print("Took %f seconds."%(time.time()-startTime))

#print("this should be 3")
#print(sess.run(tf.argmax(out,1), feed_dict={x:np.reshape(batch[0][1],[1,784]), keep_prob:1.0}))
