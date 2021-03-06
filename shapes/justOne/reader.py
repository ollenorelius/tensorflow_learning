import tensorflow as tf

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       and labels following, space separated.
    Returns:
       List with all filenames in file image_list_file,
                                        list of labels,
                                        list of coord tuples
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    coords = []

    for line in f:
        data = line.split(' ')
        filenames.append(data[0])
        labels.append(int(data[1]))
        coords.append( (float(data[2]), float(data[3])) )
    return filenames, labels, coords

def read_images_from_disk(filename):
    """Consumes a single filename.
    Args:
      filename: A scalar string tensor.
    Returns:
      One tensor: the decoded image.
    """

    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.image.resize_images(image, [128,128])
    return image

def get_justOne_batch(size,folder):

    """
        Starts the input pipe line, adds a slice_input_producer to the default graph.

        In:
            size: batch size as an integer.
            folder: path containing images to sample.

        Returns:
            tf.Tensor objects containing images, labels, coords.
    """
    image_list, label_list, coord_list = read_labeled_image_list("%s/list.txt"%folder)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    coords = tf.convert_to_tensor(coord_list, dtype=tf.float32)
    tf.reshape(labels,[-1])

    tensor_slice = tf.train.slice_input_producer([images, labels, coords], shuffle=False)

    image = read_images_from_disk(tensor_slice[0])


    image_batch, label_batch, coord_batch = tf.train.batch([image,
                                                            tensor_slice[1],
                                                            tensor_slice[2]],
                                                            batch_size=size)
    return image_batch, label_batch, coord_batch
