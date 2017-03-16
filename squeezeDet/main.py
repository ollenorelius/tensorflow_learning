import tensorflow as tf
import reader
import params
import utils as u
import kitti_reader as kr
import network as net

with tf.name_scope('Input_batching'):
    batch = reader.get_batch(40,"data/set1")
    #batch = kr.get_batch(4,"D:/KITTI")

    deltas = batch[1]
    gammas = batch[2]
    mask = batch[3]
    classes = batch[4]
    n_obj = batch[5]

    x_image = batch[0]

keep_prob = tf.placeholder(tf.float32)
#CONVOLUTIONAL LAYERS

#activations = net.create_forward_net(x_image)
activations = net.create_small_net(x_image)

with tf.name_scope('Losses'):
    with tf.name_scope('deltas'):
        d_loss = u.delta_loss(activations, deltas, mask,n_obj)
        tf.summary.scalar('Delta_loss', d_loss)
    with tf.name_scope('gammas'):
        g_loss = u.gamma_loss(activations, gammas, mask, n_obj)
        tf.summary.scalar('Gamma_loss', g_loss)
    with tf.name_scope('classes'):
        c_loss = u.class_loss(activations, classes, mask, n_obj)
        tf.summary.scalar('Class_loss', c_loss)

    loss = d_loss + g_loss + c_loss
    tf.summary.scalar('Total_loss', loss)



train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
merged = tf.summary.merge_all()

print("Model constructed!")

sess = tf.Session()

print("Variables initialized!")
import os
import sys

if not os.path.exists('./networks/'):
    os.makedirs('./networks/')


net_name = 'squeeze_small-drone-dev'
saver = tf.train.Saver()
writer = tf.summary.FileWriter("output/"+net_name, sess.graph)

coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)




if '-new' not in sys.argv:
    print('loading network.. ', end='')
    try:
        saver.restore(sess, './networks/%s.cpt'%net_name)
        print('Done.')
    except:
        print('Couldn\'t load net, creating new! E:(%s)'%sys.exc_info()[0])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


min_loss = 1000
for i in range(200000):
    if (i%1 == 0):
        d, g, c, m = sess.run([d_loss, g_loss, c_loss, merged], feed_dict = {keep_prob:1.0})
        if i > 4:
            writer.add_summary(m)
        print("step %d, d_loss: %g, g_loss: %g, c_loss: %g" % (i, d, g, c))

        if d+g+c < min_loss:
            min_loss = d+g+c
            saver.save(sess, './networks/%s.cpt'%net_name)

    sess.run(train_step, feed_dict = {keep_prob:1.0})


writer.close()

saver.save(sess, './networks/%sEND.cpt'%net_name)
