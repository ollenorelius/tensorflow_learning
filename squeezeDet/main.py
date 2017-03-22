import tensorflow as tf
import reader
import params
import utils as u
import kitti_reader as kr
import network as net

with tf.name_scope('Input_batching'):
    batch = reader.get_batch(40,["data/set1", "data/set2", "data/set3"])
    #batch = kr.get_batch(40,"/home/local-admin/KITTI")
    validation_batch = reader.get_batch(29, ["data/validation"])

    deltas = batch[1]
    gammas = batch[2]
    mask = batch[3]
    classes = batch[4]
    n_obj = batch[5]
    x_image = batch[0]

    v_deltas = validation_batch[1]
    v_gammas = validation_batch[2]
    v_mask = validation_batch[3]
    v_classes = validation_batch[4]
    v_n_obj = validation_batch[5]
    v_x_image = validation_batch[0]

#CONVOLUTIONAL LAYERS

#activations = net.create_forward_net(x_image)
#activations = net.create_small_net(x_image)
#v_activations = net.create_small_net(v_x_image,reuse=True)
activations = net.create_tiny_net(x_image)
v_activations = net.create_tiny_net(v_x_image,reuse=True)

with tf.name_scope('Losses'):
    with tf.name_scope('deltas'):
        d_loss = u.delta_loss(activations, deltas, mask,n_obj)
        tf.summary.scalar('Delta_loss_training', d_loss)
        v_d_loss = u.delta_loss(v_activations, v_deltas, v_mask, v_n_obj)
        tf.summary.scalar('Delta_loss_validation', v_d_loss)

    with tf.name_scope('gammas'):
        g_loss = u.gamma_loss(activations, gammas, mask, n_obj)
        tf.summary.scalar('Gamma_loss_training', g_loss)
        v_g_loss = u.gamma_loss(v_activations, v_gammas, v_mask, v_n_obj)
        tf.summary.scalar('Gamma_loss_validation', v_g_loss)

    with tf.name_scope('classes'):
        c_loss = u.class_loss(activations, classes, mask, n_obj)
        tf.summary.scalar('Class_loss_training', c_loss)
        v_c_loss = u.class_loss(v_activations, v_classes, v_mask, v_n_obj)
        tf.summary.scalar('Class_loss_validation', v_c_loss)

    loss = d_loss + g_loss + c_loss
    tf.summary.scalar('Total_loss_training', loss)
    v_loss = v_d_loss + v_g_loss + v_c_loss
    tf.summary.scalar('Total_loss_validation', v_loss)

with tf.name_scope('Global_step'):
    global_step = tf.Variable(0, dtype=tf.int32)

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)
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


min_loss = 100000
import time
for i in range(200000):
    start_time=time.time()
    d, g, c, m, _t = sess.run([d_loss, g_loss, c_loss, merged, train_step])
    print('training set took %f seconds!'%(time.time()-start_time))
    if (i%1 == 0):
        start_time=time.time()
        v_d, v_g, v_c, v_m = sess.run(
            [v_d_loss, v_g_loss, v_c_loss, merged])
        print('validation set took %f seconds!'%(time.time()-start_time))

        if i > 1:
            writer.add_summary(m, global_step=sess.run(global_step))

        print("step %d, d_loss: %g, g_loss: %g, c_loss: %g" % (i, d, g, c))
        print("step %d, v_d_loss: %g, v_g_loss: %g, v_c_loss: %g" % (i, v_d, v_g, v_c))

        if v_d+v_g+v_c < min_loss:
            min_loss = v_d + v_g + v_c
            start_time=time.time()
            saver.save(sess, './networks/%s.cpt'%net_name)
            print('saving took %f seconds!'%(time.time()-start_time))


writer.close()

saver.save(sess, './networks/%sEND.cpt'%net_name)
