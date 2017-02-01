from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

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

sess = tf.InteractiveSession()

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
out = tf.nn.softmax(y_conv);
print(tf.shape(y_conv))
print(tf.shape(y_))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch = mnist.train.next_batch(200)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, accuracy: %g" % (i, train_accuracy))
    train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5})

test_accuracy = accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
print("Done! accuracy on test set: %g" % (test_accuracy))

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def showData(image):
    image = np.reshape(image,(28,28))
    plt.imshow(image)
    plt.show()
    return None

img = misc.imread('2.png')
img = 1-(img[:,:,1]/255.0)
#img = np.transpose(img)/255.0
tst = np.reshape(batch[0][1],(28,28))
showData(batch[0][1])
img = np.reshape(img, [-1, 784])
showData(img)
print("this should be 2")
print(out.eval(feed_dict={x:img, keep_prob:1.0}))
print("this should be 3")
print(out.eval(feed_dict={x:np.reshape(batch[0][1],[1,784]), keep_prob:1.0}))
