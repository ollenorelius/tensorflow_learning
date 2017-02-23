import tensorflow as tf
import pickle

def open_cifar(file):
    """
        Opens a CIFAR-10 file. Takes a file name and returns a dict: [data, labels]

        data: a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
        labels: a list of 10000 numbers in the range 0-9.
    """
    fo = open(file, 'rb')
    dic = pickle.load(fo, encoding='bytes')
    fo.close()
    return dic

def get_cifar_batch(size,file):

    """
        Starts the input pipe line, adds a slice_input_producer to the default graph.

        In:
            size: batch size as an integer.
            file: path containing images to sample.

        Returns:
            tf.Tensor objects containing images, labels
    """
    import numpy as np
    dic = open_cifar(file)
    print(dic.keys())

    data, labels = dic[b'data'], dic[b'labels']

    data = np.reshape(data,[-1,3,32,32])
    data = np.swapaxes(data,1,3)

    images = tf.convert_to_tensor(data, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    tf.reshape(labels,[-1])

    tensor_slice = tf.train.slice_input_producer([images, labels], shuffle=True)



    image_batch, label_batch = tf.train.batch(tensor_slice, batch_size=size)
    return image_batch, label_batch
