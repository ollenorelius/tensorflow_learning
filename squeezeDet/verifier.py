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

batch_size = 10
with tf.name_scope('Input_batching'):
    #batch = reader.get_batch(batch_size,"data/set1")
    batch = kr.get_batch(batch_size,"/home/local-admin/KITTI")



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

x_image, deltas, gammas, mask, classes, n_obj = sess.run(batch)




#-------------------
k = p.ANCHOR_COUNT
gs = p.GRID_SIZE

#print(classes)
deltas_packed = tf.reshape(batch[1], [batch_size, gs, gs, k*p.OUT_COORDS])
gammas_packed = tf.reshape(batch[2], [batch_size, gs, gs, k])
classes_packed = tf.reshape(batch[4], [batch_size, gs, gs, k*p.OUT_CLASSES])

act_tensor_GT = tf.concat([deltas_packed, gammas_packed, 100*tf.to_float(classes_packed)], 3)

d_loss = u.delta_loss(act_tensor_GT, batch[1], batch[3], batch[5])
g_loss = u.gamma_loss(act_tensor_GT, batch[2], batch[3], batch[5])
c_loss = u.class_loss(act_tensor_GT, batch[4], batch[3], batch[5])

delta_loss = sess.run(d_loss)
gamma_loss = sess.run(g_loss)
class_loss= sess.run(c_loss)



total_loss = delta_loss + gamma_loss + class_loss

print('Autoloss for delta is: %f'%delta_loss)
print('Autoloss for gamma is: %f'%gamma_loss)
print('Autoloss for classes is: %f'%class_loss)
print('Total autoloss is: %f'%total_loss)


anchors = u.create_anchors(gs)

print(mask)

chosen_anchor = np.argmax(mask,axis=2)
for ib in range(batch_size):
    for idx in range(gs**2):
        #print(np.shape(deltas))

        ca = chosen_anchor[ib, idx]
        #print( idx+chosen_anchor[ib, idx])
        #print()
        if(gammas[ib,idx*k+ca] > 0):
            print('Box %i chosen at idx = %i with IOU: %f'%(ca,idx, gammas[ib,idx*k+ca]))
            #print(deltas[ib,ca+idx*k,:])
            #print(anchors[ca+idx*k])
            box = u.delta_to_box(deltas[ib,ca+idx*k,:],
                                anchors[ca+idx*k])
            print(u.trans_boxes(box))
