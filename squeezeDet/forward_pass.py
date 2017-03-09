import tensorflow as tf
import network as net
import utils as u
import params as p
from PIL import Image
from scipy import misc
import numpy as np

input_tensor = tf.placeholder(tf.float32, shape=[None,1242,375,3])

image = tf.image.resize_images(input_tensor, [256,256])

t_activations = net.create_small_net(image)
#t_activations = net.create_forward_net(image)
k = p.ANCHOR_COUNT
t_deltas = tf.slice(t_activations, [0,0,0,0], [-1,-1,-1,4*k])
t_gammas = tf.sigmoid(tf.slice(t_activations, [0,0,0,4*k], [-1,-1,-1,k]))
t_classes = tf.slice(t_activations, [0,0,0,5*k], [-1,-1,-1,p.OUT_CLASSES*k])

t_chosen_anchor = tf.argmax(t_gammas, axis=3)

all_out = [t_activations, t_deltas, t_gammas, t_classes, t_chosen_anchor]

sess = tf.Session()
batch_size = 1
print('loading image.. ', end='')
img = [np.transpose(misc.imread('000001.png'),[1, 0, 2])]
print('Done.')
net_name = 'squeasdfezeKITTI_dev'
print('loading network.. ', end='')
try:
    saver = tf.train.Saver()
    saver.restore(sess, './networks/%s.cpt'%net_name)
    print('Done.')
except:
    print('Failed! using random net.')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

import time

start_time = time.time()
activations, deltas, gammas, classes, chosen_anchor = \
                sess.run(all_out, feed_dict={input_tensor: img})
print('Took %f seconds!'%(time.time()-start_time))


k = p.ANCHOR_COUNT
gs = p.GRID_SIZE

#print(gammas)

gammas = np.reshape(gammas, [-1, gs**2*k])
chosen_anchor = np.reshape(chosen_anchor,[-1,gs**2])
deltas = np.reshape(deltas, [-1, gs**2*k,4])
anchors = u.create_anchors(gs)
classes = np.reshape(classes, [-1,gs**2*k, p.OUT_CLASSES])
class_numbers = np.argmax(classes, axis=2)

for ib in range(batch_size):
    #pic = Image.fromarray(img[ib])
    max_gamma= 0
    for idx in range(gs**2):
        #print(np.shape(gammas))

        ca = chosen_anchor[ib, idx]
        #print( idx+chosen_anchor[ib, idx])
        #print(chosen_anchor.shape)
        if(gammas[ib,idx*k+ca] > 0):
            print('Anchor %i chosen at idx = %i for class %i with conf: %f'\
                        %(ca,idx,class_numbers[ib,idx*k+ca], gammas[ib,idx*k+ca]))
            box = u.delta_to_box(deltas[ib,ca+idx*k,:],
                                anchors[ca+idx*k])
            print(u.trans_boxes(box))
            if(gammas[ib,idx*k+ca] > max_gamma):
                max_gamma = gammas[ib,idx*k+ca]
                print('BEST')
