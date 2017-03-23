import tensorflow as tf
import network as net
import utils as u
import params as p
from PIL import Image
from scipy import misc
import numpy as np
from tensorflow.python.platform import gfile


net_name = 'squeeze_normal-drone-dev'
folder_name = './networks/%s'%net_name
with gfile.FastGFile(folder_name + "/minimal_graph_quant.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

sq_graph = tf.get_default_graph()
inp_batch = sq_graph.get_tensor_by_name('Input_batching/batch:0')
t_activations = sq_graph.get_tensor_by_name('activation/activations:0')
print(inp_batch)

k = p.ANCHOR_COUNT
t_deltas = tf.slice(t_activations, [0,0,0,0], [-1,-1,-1,4*k])
t_gammas = tf.sigmoid(tf.slice(t_activations, [0,0,0,4*k], [-1,-1,-1,k]))
t_classes = tf.slice(t_activations, [0,0,0,5*k], [-1,-1,-1,p.OUT_CLASSES*k])

t_chosen_anchor = tf.argmax(t_gammas, axis=3)

all_out = [t_deltas, t_gammas, t_classes, t_chosen_anchor]

sess = tf.Session()
batch_size = 1
print('loading image.. ', end='')

def read_resize(pic):
    return misc.imresize(misc.imread(pic), (256,256))

img = [read_resize('/home/local-admin/KITTI/training/image_2/000001.png'),
read_resize('/home/local-admin/KITTI/training/image_2/000002.png'),
read_resize('/home/local-admin/KITTI/training/image_2/000003.png')]

print('Done.')

import time

for i in range(5):
    start_time = time.time()
    deltas, gammas, classes, chosen_anchor = \
                    sess.run(all_out, feed_dict={inp_batch: img})
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
    print('image %s'%ib)
    for idx in range(gs**2):
        #print(np.shape(gammas))

        ca = chosen_anchor[ib, idx]
        #print( idx+chosen_anchor[ib, idx])
        #print(chosen_anchor.shape)
        if(gammas[ib,idx*k+ca] > 0.05):
            #print('Anchor %i chosen at idx = %i for class %i with conf: %f'\
            #            %(ca,idx,class_numbers[ib,idx*k+ca], gammas[ib,idx*k+ca]))
            box = u.delta_to_box(deltas[ib,ca+idx*k,:],
                                anchors[ca+idx*k])
            #print(u.trans_boxes(box))
            if(gammas[ib,idx*k+ca] > max_gamma):
                max_gamma = gammas[ib,idx*k+ca]
                print('BEST')
