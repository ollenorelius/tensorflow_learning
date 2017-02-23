import tensorflow as tf
import reader
import params
import utils as u


batch = reader.get_cifar_batch(300,"cifar-10-batches-py/data_batch_1")

oh = tf.one_hot(batch[1], params.OUT_CLASSES, dtype=tf.int32)

x_image = batch[0]
y_ = oh
keep_prob = tf.placeholder(tf.float32)
#CONVOLUTIONAL LAYERS
x_image = tf.reshape(x_image, [-1, 32,32,3])
sq1 = u.create_fire_module(x_image,16,64,64,3)
mp1 = u.max_pool_2x2(sq1) #down to 16x16

sq2 = u.create_fire_module(mp1, 16,64,64,128)
sq3 = u.create_fire_module(sq2, 16,64,64,128)
sq4 = u.create_fire_module(sq3, 32,128,128,128)

mp4 = u.max_pool_2x2(sq4) #down to 8x8

#sq5 = u.create_fire_module(mp4, 32,128,128,256)
#sq6 = u.create_fire_module(sq5, 48,192,192,256)
#sq7 = u.create_fire_module(sq6, 48,192,192,384)
#sq8 = u.create_fire_module(sq7, 64,256,256,384)

#mp8 = u.max_pool_2x2(sq8)#down to 4x4

sq9 = u.create_fire_module(mp4, 32,128,128,256)#(mp8, 64,256,256,512)

activations = u.get_activations(sq9, 8, 256)

out = tf.nn.softmax(activations)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activations,y_))
loss = cross_entropy

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print("Model constructed!")

sess = tf.Session()

coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())



print("Variables initialized!")
import os
saver = tf.train.Saver()
for i in range(200000):
    if i%30 == 0:
        train_accuracy = sess.run(accuracy, feed_dict = {keep_prob:1.0})
        ce = sess.run(cross_entropy, feed_dict = {keep_prob:1.0})
        print("step %d, class accuracy: %g, cross entropy: %g" % (i, train_accuracy, ce))

    if i%200 == 0:
        saver.save(sess, './networks/squeezeNight.cpt')
    sess.run(train_step, feed_dict = {keep_prob:1.0})


test_accuracy = sess.run(accuracy, feed_dict = {keep_prob:1.0})

print("Done! accuracy on test set: %g" % (test_accuracy))


if not os.path.exists('./networks/'):
    os.makedirs('./networks/')
saver.save(sess, './networks/squeezenightEND.cpt')
