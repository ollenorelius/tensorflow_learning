import tensorflow as tf
import params as p

def create_fire_module(input_tensor,s1,e1,e3, in_channels):
    """
        Adds a Fire module to the graph.

        In:
            input_tensor: the preceding tf.Tensor.
            s1: number of 1x1 squeeze kernels.
            e1: number of 1x1 expand kernels.
            e3: number of 3x3 expand kernels.

        Out:
            tens: Activation volume (tf.Tensor)
    """

    tens = squeeze(input_tensor,s1,in_channels)
    tens = expand(tens,e1,e3,s1)
    return tens


def squeeze(input_tensor, s1, in_channels):
    """
        Creates a squeeze operation on input_tensor.
        In:
            input_tensor: the preceding tf.Tensor.
            s1: number of 1x1 kernels.

        Out:
            Activation volume. (tf.Tensor)
    """

    w = weight_variable([1,1,in_channels,s1])
    b = bias_variable([s1])
    return tf.nn.relu(conv2d(input_tensor, w) + b)


def expand(input_tensor, e1, e3, in_channels):
    """
        Creates a expand operation on input_tensor.
        In:
            input_tensor: the preceding tf.Tensor.
            e1: number of 1x1 kernels.
            e3: number of 3x3 kernels.

        Out:
            Activation volume. (tf.Tensor)
    """

    w3 = weight_variable([3,3,in_channels,e3])
    b3 = bias_variable([e3])
    c3 = tf.nn.relu(conv2d(input_tensor, w3) + b3)

    w1 = weight_variable([1,1,in_channels,e1])
    b1 = bias_variable([e1])
    c1 = tf.nn.relu(conv2d(input_tensor, w1) + b1)

    return tf.concat(3,[c1,c3])


def get_activations(input_tensor, in_size, in_channels):
    """
        Gets activations by 1x1 convolution and avg pooling as described in the
        SqueezeNet paper.

        In:
            Activation volume from previous convolutional layers. (tf.Tensor)

        Out:
            tf.Tensor of class scores. (batch x classes)
    """

    w = weight_variable([1, 1, in_channels, p.OUT_CLASSES])
    b = bias_variable([p.OUT_CLASSES])
    tens = tf.nn.relu(conv2d(input_tensor, w) + b)


    act = tf.nn.avg_pool(tens,
                            ksize=[1, in_size, in_size, 1],
                            strides=[1,1,1,1],
                            padding='VALID')

    return tf.squeeze(act)


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
