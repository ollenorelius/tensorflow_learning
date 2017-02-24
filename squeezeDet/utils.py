import tensorflow as tf
import params as p
import numpy as np

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
    out_count = p.OUT_CLASSES + p.OUT_COORDS + p.OUT_CONF
    w = weight_variable([3, 3, in_channels, p.OUT_CLASSES])
    b = bias_variable([p.OUT_CLASSES])
    tens = tf.nn.relu(conv2d(input_tensor, w) + b)

    return tens

def create_anchors(grid_size):
    """
        Creates a list of all anchor coordinates.
        Input: Grid size to distribute anchors over.

        Returns: (K*X*Y) x 4 tensor containing coordinates in (x,y,w,h)-form
            in a, x, y order:
    """
    x = np.linspace(0,1,grid_size)
    xv, yv = np.meshgrid(x,x)
    anchors = []
    for iy in range(len(xv)):
        for ix in range(len(yv)):
            for i in range(p.ANCHOR_COUNT):
                anchors.append((xv[ix,iy], yv[ix,iy], p.ANCHOR_SIZES[i][0], p.ANCHOR_SIZES[i][1]))
    assert (len(anchors), len(anchors[1])) == (p.ANCHOR_COUNT * p.GRID_SIZE**2, 4), \
     "ERROR: create_anchors made a matrix of shape %i,%i" % (len(anchors), len(anchors[1]))

    return anchors

def intersection(bbox, anchors):
    """
    Computes intersection of a SINGLE bounding box and all anchors.

    bbox: coordinate tensor: 4x1 (x1, y1, x2, y2)
    anchors: coordinate tensor: 4x(X*Y*K) with XY being number of grid points

    returns: a 1x(X*Y*K) tensor containing all anchor intersections.

    """


    p1 = np.minimum(bbox[2], anchors[2]) - np.maximum(bbox[0], anchors[0])
    p2 = np.minimum(bbox[3], anchors[3]) - np.maximum(bbox[1], anchors[1])

    p1_r = np.maximum(p1,0) #If this is negative, there is no intersection
    p2_r = np.maximum(p2,0) # so it is rectified

    return np.multiply(p1_r,p2_r)

def union(bbox, anchors, intersections):
    """
    Computes union of a SINGLE bounding box and all anchors.

    bbox: coordinate array: 4x1 (x1, y1, x2, y2)
    anchors: coordinate array: 4x(X*Y*K) with XY being number of grid points
    intersections: array containing all intersections computed using
        intersection(). used to avoid double calculation.


    returns: a 1x(X*Y*K) array containing all anchor unions.

    """

    box_area = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])
    anchor_areas = (anchors[2] - anchors[0])*(anchors[3] - anchors[1])

    return box_area + anchor_areas - intersections

def intersection_over_union(bbox, anchors):
    intersections = intersection(bbox, anchors)
    unions = union(bbox, anchors, intersections)
    return tf.divide(intersections, unions)

def trans_boxes(coords):
    """
        Transforms coordinates from x, y, w, h to x1, y1, x2, y2.

        Input: a Nx4 matrix of coordinates for N boxes.

        Returns: a Nx4 matrix of transformed coordinates.
    """
    coords = np.transpose(coords)
    t_coords = []
    t_coords.append(coords[0] - coords[2]/2)
    t_coords.append(coords[0] + coords[2]/2)
    t_coords.append(coords[1] - coords[3]/2)
    t_coords.append(coords[1] + coords[3]/2)
    t_coords = np.transpose(t_coords)
    return t_coords

def inv_trans_boxes(coords):
    """
        Transforms coordinates from x1, y1, x2, y2 to x, y, w, h

        Input: a Nx4 matrix of coordinates for N boxes.

        Returns: a Nx4 matrix of transformed coordinates.
    """
    coords = np.transpose(coords)
    t_coords = []
    t_coords.append((coords[0] + coords[2])/2)
    t_coords.append((coords[1] + coords[3])/2)
    t_coords.append(coords[2] - coords[0])
    t_coords.append(coords[3] - coords[1])
    t_coords = np.transpose(t_coords)
    assert np.shape(t_coords)[1] == 4, \
            "invalid shape in inv_trans_boxes: %i,%i" % np.shape(t_coords)

    return t_coords



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