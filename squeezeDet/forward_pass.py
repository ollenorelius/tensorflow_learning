import tensorflow as tf
import network as net
import utils as u
import params as p
from PIL import Image
from scipy import misc


input_tensor = tf.placeholder(tf.float32, shape=[None,375,1242,3])

image = tf.image.resize_images(input_tensor, [256,256])
t_activations = net.create_forward_net(image)

k = p.ANCHOR_COUNT
t_deltas = tf.slice(t_activations, [0,0,0,0], [-1,-1,-1,4*k])
t_gammas = tf.sigmoid(tf.slice(t_activations, [0,0,0,4*k], [-1,-1,-1,k]))
t_classes = tf.slice(t_activations, [0,0,0,5*k], [-1,-1,-1,p.OUT_CLASSES*k])

t_chosen_anchor = tf.argmax(t_gammas, axis=3)

all_out = [t_activations, t_deltas, t_gammas, t_classes, t_chosen_anchor]

sess = tf.Session()
batch_size = 1
print('loading image.. ', end='')
img = [misc.imread('/home/local-admin/KITTI/training/image_2/000001.png')]
print('Done.')

print('loading network.. ', end='')
saver = tf.train.Saver()
saver.restore(sess, './networks/squeezeKITTI.cpt')
print('Done.')
sess.run(tf.global_variables_initializer())

activations, deltas, gammas, classes, chosen_anchor = \
                sess.run(all_out, feed_dict={input_tensor: img})

anchors = u.create_anchors(p.GRID_SIZE)
anchor_index = 0
for ib in range(batch_size):
    pic = Image.fromarray(img[ib])
    for iy in range(p.GRID_SIZE):
        for ix in range(p.GRID_SIZE):
            print(gammas[ib,ix,iy,chosen_anchor[ib,ix,iy]])
            if(gammas[ib,ix,iy,chosen_anchor[ib,ix,iy]] > 0.1):
                a_a = chosen_anchor[ib,ix,iy]
                a_i = anchor_index*p.ANCHOR_COUNT
                box = u.delta_to_box(deltas[ib,ix,iy,:],
                                    anchors[a_a+a_i])
                print(box)
