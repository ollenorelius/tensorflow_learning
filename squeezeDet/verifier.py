import tensorflow as tf
import reader
import params as p
import utils as u
import kitti_reader as kr
import network as net
import numpy as np

"""
Script for unit testing my delta and gamma calculations.
I load the boxes from the first picture in the batch and then
try to get display boxes from those deltas.
"""


with tf.name_scope('Input_batching'):
    #batch = reader.get_batch(1,"data/set1")
    batch = kr.get_batch(1,"data")



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




#-------------------
k = p.ANCHOR_COUNT
gs = p.GRID_SIZE

anchors = u.create_anchors(gs)

#print(anchors)
batch_size = 1
chosen_anchor = np.argmax(mask,axis=2)
for ib in range(batch_size):
    for idx in range(gs**2):
        #print(np.shape(deltas))

        ca = chosen_anchor[ib, idx]
        #print( idx+chosen_anchor[ib, idx])
        #print()
        if(gammas[ib,idx*k+ca] > 0):
            print('Box %i chosen at idx = %i with IOU: %f'%(ca,idx, gammas[ib,idx*k+ca]))
            box = u.delta_to_box(deltas[ib,ca+idx*k,:],
                                anchors[ca+idx*k])
            print(u.trans_boxes(box))
