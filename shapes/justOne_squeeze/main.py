import tensorflow as tf
import reader
import params
import utils as u


batch = reader.get_justOne_batch(300,"training")
oh = tf.one_hot(batch[1], params.OUT_CLASSES, dtype=tf.int32)

x_image = batch[0]
y_ = oh
coords_ = batch[2]

#CONVOLUTIONAL LAYERS
x_image = tf.reshape(x_image, [-1, 128,128,3])
sq1 = u.create_fire_module(x_image,16,64,64,3)
mp1 = u.max_pool_2x2(sq1) #down to 64x64

sq2 = u.create_fire_module(mp1, 16,64,64,128)
sq3 = u.create_fire_module(sq2, 16,64,64,128)
sq4 = u.create_fire_module(sq3, 32,128,128,128)

mp4 = u.max_pool_2x2(sq4) #down to 32x32

sq5 = u.create_fire_module(mp4, 32,128,128,256)
sq6 = u.create_fire_module(sq5, 48,192,192,256)
sq7 = u.create_fire_module(sq6, 48,192,192,384)
sq8 = u.create_fire_module(sq7, 64,256,256,384)

mp8 = u.max_pool_2x2(sq8)#down to 16x16

sq9 = u.create_fire_module(mp8, 64,256,256,512)

activations = u.get_activations(sq9, 16, 512)

out = tf.nn.softmax(activations)

#Regressor

keep_prob = tf.placeholder(tf.float32)
final_count = tf.cast((params.IMAGE_SIZE/4)**2*512,tf.int32)
h_sq8_flat = tf.reshape(sq8,[-1, final_count])

W_reg1 = u.weight_variable([final_count, params.FC_NODES])
b_reg1 = u.bias_variable([params.FC_NODES])

h_reg1 = tf.nn.relu(tf.matmul(h_sq8_flat, W_reg1) + b_reg1)
h_reg1_drop = tf.nn.dropout(h_reg1, keep_prob)

W_reg2 = u.weight_variable([params.FC_NODES, 2])
b_reg2 = u.bias_variable([2])

coord_predict = tf.sigmoid(tf.matmul(h_reg1_drop, W_reg2) + b_reg2)

regression_loss = tf.nn.l2_loss(coords_ - coord_predict)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activations,y_))
loss = cross_entropy + regression_loss

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print("Model constructed!")

sess = tf.Session()

coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())



print("Variables initialized!")

for i in range(200):
    if i%1 == 0:

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
saver.save(sess, './networks/squeeze1.cpt')
