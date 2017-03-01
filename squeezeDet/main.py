import tensorflow as tf
import reader
import params
import utils as u


batch = reader.get_batch(40,"data/set1")

oh = tf.one_hot(batch[1], params.OUT_CLASSES, dtype=tf.int32)

coord = batch[2]
deltas = batch[3]
gammas = batch[4]
mask = batch[5]
classes = batch[6]

x_image = batch[0]
y_ = oh
keep_prob = tf.placeholder(tf.float32)
#CONVOLUTIONAL LAYERS
x_image = tf.reshape(x_image, [-1, 256,256,3])
sq1 = u.create_fire_module(x_image,32,64,64)
mp1 = u.max_pool_2x2(sq1) #down to 128x128

sq2 = u.create_fire_module(mp1, 16,64,64)
sq3 = u.create_fire_module(sq2, 16,64,64)
sq4 = u.create_fire_module(sq3, 32,128,128)

mp2 = u.max_pool_2x2(sq4) # 64x64
mp3 = u.max_pool_2x2(mp2) #down to 32x32

sq5 = u.create_fire_module(mp3, 32,128,128)
sq6 = u.create_fire_module(sq5, 48,192,192)
sq7 = u.create_fire_module(sq6, 48,192,192)
sq8 = u.create_fire_module(sq7, 64,256,256)

mp4 = u.max_pool_2x2(sq8)#down to 16x16
mp5 = u.max_pool_2x2(mp4) # 8x8


sq9 = u.create_fire_module(mp5, 64,256,256)#(mp8, 64,256,256,512)

activations = u.get_activations(sq9)

loss = u.loss_function(activations, deltas, gammas, mask, classes)



train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


print("Model constructed!")

sess = tf.Session()

coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())



print("Variables initialized!")
import os

if not os.path.exists('./networks/'):
    os.makedirs('./networks/')

saver = tf.train.Saver()
for i in range(200000):
    if i%1 == 0:
        train_accuracy = sess.run(loss, feed_dict = {keep_prob:1.0})
        print("step %d, loss: %g" % (i, train_accuracy))

    if i%200 == 0:
        saver.save(sess, './networks/squeezeNight.cpt')
    sess.run(train_step, feed_dict = {keep_prob:1.0})


test_accuracy = sess.run(accuracy, feed_dict = {keep_prob:1.0})

print("Done! accuracy on test set: %g" % (test_accuracy))


saver.save(sess, './networks/squeezenightEND.cpt')
