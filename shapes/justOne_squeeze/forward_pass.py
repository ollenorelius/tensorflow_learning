import tensorflow as tf
import reader
import params

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

x_image = tf.placeholder(tf.float32, shape=[None, params.IMAGE_SIZE, params.IMAGE_SIZE, 3])

ks = params.CN_KERN_SIZE
kc = params.CN_KERN_COUNT

#CONVOLUTIONAL LAYERS
W_conv1 = weight_variable([ks,ks,3,kc])
b_conv1 = bias_variable([kc])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([ks,ks,kc,kc])
b_conv2 = bias_variable([kc])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

final_count = tf.cast((params.IMAGE_SIZE/4)**2*kc,tf.int32)
h_pool2_flat = tf.reshape(h_pool2,[-1, final_count])

#Fully connected classifier
W_fc1 = weight_variable([final_count, params.FC_NODES])
b_fc1 = bias_variable([params.FC_NODES])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([params.FC_NODES, params.OUT_CLASSES])
b_fc2 = bias_variable([params.OUT_CLASSES])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
out = tf.nn.softmax(y_conv)

#Regressor

W_reg1 = weight_variable([final_count, params.FC_NODES])
b_reg1 = bias_variable([params.FC_NODES])

h_reg1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_reg1) + b_reg1)
h_reg1_drop = tf.nn.dropout(h_reg1, keep_prob)

W_reg2 = weight_variable([params.FC_NODES, 2])
b_reg2 = bias_variable([2])

coord_predict = tf.sigmoid(tf.matmul(h_reg1_drop, W_reg2) + b_reg2)

print("Model constructed!")

sess = tf.Session()

coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())



print("Variables initialized!")

from scipy import misc
import numpy as np

print("Loading network...", end=" ")
saver = tf.train.Saver()
saver.restore(sess, './networks/straightPipe1.cpt')
print("Done!")
#batch = mnist.train.next_batch(200)

pic = misc.imread('test/2.jpg')
pic = np.reshape(pic,(1,128,128,3))
bSize = 1
pic = np.repeat(pic, bSize, 0)
import time
print("Started")
startTime = time.time()
runs = 10
for i in range(runs):
    [predicted_class, predicted_coord] = sess.run([out, coord_predict], feed_dict = {x_image:pic, keep_prob:1.0})
    #predicted_coord = sess.run(, feed_dict = {x_image:pic, keep_prob:1.0})


print(predicted_coord)
print("Done! Class = %g, coords: (%g,%g)" % (np.argmax(predicted_class), predicted_coord[0][0],predicted_coord[0][1]))
print("Took %g secs per pass."%((time.time()-startTime)/(runs*bSize)))

import os
saver = tf.train.Saver()
if not os.path.exists('./networks/'):
    os.makedirs('./networks/')
saver.save(sess, './networks/straightPipe1.cpt')
