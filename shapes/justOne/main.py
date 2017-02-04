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

#Input pipeline in reader.py
batch = reader.get_justOne_batch(300,"training")
oh = tf.one_hot(batch[1], params.OUT_CLASSES, dtype=tf.int32)

x_image = batch[0]
y_ = oh
coords_ = batch[2]

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

regression_loss = tf.nn.l2_loss(coords_ - coord_predict)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))
loss = cross_entropy + regression_loss

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print("Model constructed!")

sess = tf.Session()

coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())



print("Variables initialized!")

for i in range(200):
    if i%5 == 0:

        train_accuracy = sess.run(accuracy, feed_dict = {keep_prob:1.0})
        pos_acc = sess.run(regression_loss, feed_dict = {keep_prob:1.0})
        print("step %d, class accuracy: %g, position loss: %g" % (i, train_accuracy, pos_acc))

    sess.run(train_step, feed_dict = {keep_prob:1.0})


test_accuracy = sess.run(accuracy, feed_dict = {keep_prob:1.0})

print("Done! accuracy on test set: %g" % (test_accuracy))

import os
saver = tf.train.Saver()
if not os.path.exists('./networks/'):
    os.makedirs('./networks/')
saver.save(sess, './networks/straightPipe1.cpt')
