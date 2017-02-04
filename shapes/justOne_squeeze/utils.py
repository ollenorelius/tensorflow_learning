import tensorflow as tf
import params as p

def create_fire_module(input_tensor, (s1,e1,e3)):
    tens = squeeze(input_tensor,s1)
    tens = expand(tens,e1,e3)
    return tens


def squeeze(input_tensor, s1):
    nc = tf.shape(input_tensor)[3]
    w = weight_variable([1,1,nc,s1])
    b = bias_variable([s1])
    return tf.nn.relu(conv2d(input_tensor, w) + b)


def expand(input_tensor, e1, e3):
    nc = tf.shape(input_tensor)[3]

    w3 = weight_variable([3,3,nc,e3])
    b3 = bias_variable([e3])
    c3 = tf.nn.relu(conv2d(input_tensor, w3) + b3)

    w1 = weight_variable([1,1,nc,e1])
    b1 = bias_variable([e1])
    c1 = tf.nn.relu(conv2d(input_tensor, w1) + b1)

    return tf.concat(0,[c1,c3])


def get_activations(input_tensor):
    w = weight_variable([1,1,p.OUT_CLASSES])
    b = bias_variable([p.OUT_CLASSES])
    tens = tf.nn.relu(conv2d(input_tensor, w) + b)
    sh = tf.shape(input_tensor)


    act = tf.nn.avg_pool(tens,
                            ksize=[1, sh[1], sh[1], 1],
                            strides=[1,1,1,1],
                            padding='SAME')

    return act


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
