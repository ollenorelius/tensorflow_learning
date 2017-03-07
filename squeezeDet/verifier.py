import tensorflow as tf
import reader
import params
import utils as u
import kitti_reader as kr
import network as net
import numpy as np

with tf.name_scope('Input_batching'):
#batch = reader.get_batch(40,"data/set1")
    batch = kr.get_batch(4,"data")



keep_prob = tf.placeholder(tf.float32)
#CONVOLUTIONAL LAYERS


print("Model constructed!")

sess = tf.Session()

print("Variables initialized!")
import os
import sys

if not os.path.exists('./networks/'):
    os.makedirs('./networks/')



coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

deltas = sess.run(batch[1])
gammas = sess.run(batch[2])
mask = sess.run(batch[3])
classes = sess.run(batch[4])
n_obj = sess.run(batch[5])

x_image = sess.run(batch[0])



anchors = u.create_anchors(params.GRID_SIZE)
print(anchors)
batch_size = 1
chosen_anchor = np.argmax(mask,axis=2)
for ib in range(batch_size):
    for ix in range(params.GRID_SIZE):
        for iy in range(params.GRID_SIZE):
            print(np.shape(anchors))
            I = (iy*params.GRID_SIZE+ix)*params.ANCHOR_COUNT
            ca = chosen_anchor[ib,iy*params.GRID_SIZE+ix]
            print(I+chosen_anchor[ib,iy*params.GRID_SIZE+ix])
            #print(gammas)
            if(gammas[ib,I+ca] > 0):
                box = u.delta_to_box(deltas[ib,I,:],
                                    anchors[ca+I])
                print(box)
