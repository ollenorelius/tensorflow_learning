"""Main training script for SqueezeDet."""
import tensorflow as tf
import reader
import params
import utils as u
import network as net
import os
import sys
import time

with tf.variable_scope('Input_batching'):
    data_folders = ["data/training"]
    validation_folders = ['data/test']

    with tf.device('/cpu:0'):
        batch = reader.get_batch(params.BATCH_SIZE, data_folders)
        # batch = kr.get_batch(40,"/home/local-admin/KITTI")
        validation_batch = reader.get_batch(params.BATCH_SIZE,
                                            validation_folders)

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

dropout = tf.placeholder(tf.float32)

# CONVOLUTIONAL LAYERS
# feature_map = net.create_forward_net_new(x_image, dropout)
# v_feature_map = net.create_forward_net_new(v_x_image, dropout, reuse=True)

feature_map = net.create_tiny_net_res(x_image, dropout)
v_feature_map = net.create_tiny_net_res(v_x_image, dropout, reuse=True)

activations = net.get_activations(feature_map, 'activations')
v_activations = net.get_activations(v_feature_map,
                                    'valid_activations',
                                    reuse=True)

tf.summary.histogram('activations', activations)
tf.summary.histogram('validation_activations', v_activations)

with tf.device('/cpu:0'):
    with tf.name_scope('Losses'):
        with tf.name_scope('deltas'):
            d_loss = u.delta_loss(activations, deltas, mask, n_obj)
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

        t_vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in t_vars]) * 0.001
        loss = d_loss + g_loss + c_loss + lossL2

        tf.summary.scalar('Total_loss_training', loss)
        v_loss = v_d_loss + v_g_loss + v_c_loss
        tf.summary.scalar('Total_loss_validation', v_loss)

with tf.name_scope('Global_step'):
    global_step = tf.Variable(0, dtype=tf.int32)

train_step = tf.train.AdamOptimizer(0.001).minimize(loss,
                                                    global_step=global_step)

merged = tf.summary.merge_all()

print("Model constructed!")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print("Variables initialized!")


net_name = 'tiny_res'
folder_name = './networks/%s' % net_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

saver = tf.train.Saver()
writer = tf.summary.FileWriter("output/"+net_name, sess.graph)
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)


if '-new' not in sys.argv:
    print('loading network.. ', end='')
    try:
        if '-best' in sys.argv:
            saver.restore(sess, folder_name + '/best_valid.cpt')
            print('Starting from best net.. ', end='')
            print('Done.')
        else:
            saver.restore(sess, folder_name + '/latest.cpt')
            print('Starting from latest net.. ', end='')
            print('Done.')
    except:
        print('Couldn\'t load net, creating new! E:(%s)' % sys.exc_info()[0])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

min_loss = 100
# print([n.name for n in tf.get_default_graph().as_graph_def().node])

with sess.as_default():
    for i in range(20000000):
        start_time = time.time()
        d, g, c, m, _t = sess.run([d_loss, g_loss, c_loss, merged, train_step],
                                  feed_dict={dropout: 0.5})
        t = time.time()-start_time
        print('training set took %f seconds! (%s FPS)'
              % (t, params.BATCH_SIZE/t), end='\r')
        if (i % 10 == 0):
            start_time = time.time()
            v_d, v_g, v_c, v_m = sess.run(
                [v_d_loss, v_g_loss, v_c_loss, merged],
                feed_dict={dropout: 1.0})

            print('validation set took %f seconds!' % (time.time()-start_time))

            if i > -1:
                writer.add_summary(m, global_step=sess.run(global_step))

            print("step %d, d_loss: %g, g_loss: %g, c_loss: %g"
                  % (i, d, g, c))
            print("step %d, v_d_loss: %g, v_g_loss: %g, v_c_loss: %g"
                  % (i, v_d, v_g, v_c))

            if (i % 1000 == 0):
                saver.save(sess, folder_name + '/latest.cpt')
                # write graph to protobuf and then quantize
                u.write_graph_to_pb(sess,
                                    'activation/activations',
                                    'latest',
                                    folder_name)

            if v_d+v_g+v_c < min_loss:
                min_loss = v_d + v_g + v_c
                start_time = time.time()
                saver.save(sess, folder_name + '/best_valid.cpt')

                # write graph to protobuf and then quantize
                u.write_graph_to_pb(sess,
                                    'activation/activations',
                                    'best_valid',
                                    folder_name)
                print('saving took %f seconds!' % (time.time()-start_time))


writer.close()

saver.save(sess, './networks/%sEND.cpt' % net_name)
