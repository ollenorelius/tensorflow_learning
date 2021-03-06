"""All functions related to reading files and processing data."""
import tensorflow as tf
import params as p
import utils as u
import numpy as np
import os
import re


def read_image_folders(folders, classes=None):
    """Read a .txt file containing paths and labels.

    Args:
       image_list_file: a .txt file with one /path/to/image
            followed by 4 coords [x1 y1 x2 y2],
            followed by a integer class per line.
        classes: A list of classes that will be read from file.
            If not set, or set to None, all classes in file will be chosen.

    Returns:

        filenames: List with all filenames in file image_list_file.
        N_obj: vector of number of objects in each image. [images, 1]
        ret_deltas: 3D matrix of deltas for every image and grid point.
                            [images, gs**2*k, 4]
        ret_gammas: matrix of gammas for each image and grid point.
                            [images, gs**2*k,1],
        ret_masks:  mask highlighting the highest IOU for every grid point.
                            [images, gs**2*k,1]
        ret_classes: matrix of one-hot vectors.
                            [images, gs**2*k, classes]
    """
    if type(folders) == str:
        folders = [folders]

    labels = []
    coords = []
    filename_list = []  # This is a list built from the names found in list.txt
    ret_deltas = []
    ret_gammas = []
    ret_masks = []
    ret_classes = []
    N_obj = []
    filenames = []  # This is all of the files in the folder
    anchors = np.asarray(u.create_anchors(p.GRID_SIZE))  # KXY x 4, (x,y,w,h)

    for folder in folders:
        filenames_unfiltered = os.listdir(folder)
        for unf in filenames_unfiltered:
            if re.search('\.jpg\Z', unf) is not None:
                filenames.append(folder + '/' + unf)

        f = open(folder + '/list.txt', 'r')
        for line in f:
            data = line.split(' ')
            data[5] = int(data[5])-1  # TEMP classes from {1,2} to {0,1}
            if classes is None or int(data[5]) in classes:  # Class filtering.
                fn = folder + '/' + data[0]
                if fn in filename_list:
                    ind = filename_list.index(fn)
                    labels[ind].append(int(data[5]))
                    coords[ind].append((float(data[1]),
                                        float(data[2]),
                                        float(data[3]),
                                        float(data[4])))
                else:
                    filename_list.append(fn)
                    labels.append([int(data[5])])
                    coords.append([(float(data[1]),
                                    float(data[2]),
                                    float(data[3]),
                                    float(data[4]))])

    for i in range(len(filename_list)):

        # which box is each anchor assigned to?
        box_mask = np.zeros([p.GRID_SIZE * p.GRID_SIZE * p.ANCHOR_COUNT])
        # This is I_ijk in the paper
        input_mask = np.zeros([p.GRID_SIZE * p.GRID_SIZE * p.ANCHOR_COUNT])
        # What is the IOU at every grid point and every anchor?
        iou_mask = np.zeros([p.GRID_SIZE * p.GRID_SIZE * p.ANCHOR_COUNT])
        deltas = np.zeros([p.GRID_SIZE*p.GRID_SIZE*p.ANCHOR_COUNT, 4])
        it_coords = u.inv_trans_boxes(coords[i])  # x,y,w,h
        classes = np.zeros([p.GRID_SIZE * p.GRID_SIZE * p.ANCHOR_COUNT,
                            p.OUT_CLASSES])

        ious = []
        for box in it_coords:
            ious.append(u.intersection_over_union(box, np.transpose(anchors)))
        ious = np.array(ious)  # N_obj x XYK

        box_mask = np.argmax(ious, 0)  # XYK x 1
        iou_mask = np.amax(ious, 0)  # XYK x 1

        box_assignment = np.argmax(ious, 1)

        chosen_boxes = it_coords[box_mask, :]

        xg = chosen_boxes[:, 0]
        yg = chosen_boxes[:, 1]
        wg = chosen_boxes[:, 2]
        hg = chosen_boxes[:, 3]

        x_hat = anchors[:, 0]
        y_hat = anchors[:, 1]
        w_hat = anchors[:, 2]
        h_hat = anchors[:, 3]

        deltas[:, 0] = (xg-x_hat)/w_hat
        deltas[:, 1] = (yg-y_hat)/h_hat
        deltas[:, 2] = np.log(wg/w_hat)
        deltas[:, 3] = np.log(hg/h_hat)

        # Reshaping the IOUs to be a matrix of [grid points x anchors]
        input_mask = np.zeros([p.GRID_SIZE**2 * p.ANCHOR_COUNT])
        input_mask[box_assignment] = 1

        for j in range(p.GRID_SIZE**2*p.ANCHOR_COUNT):
            classes[j, labels[i][box_mask[j]]] = 1

        ret_deltas.append(np.nan_to_num(deltas))
        ret_gammas.append(iou_mask)
        ret_masks.append(input_mask)
        ret_classes.append(classes)
        N_obj.append(len(labels[i]))

    '''for picture in filenames:
        if picture not in filename_list:
            filename_list.append(picture)
            labels.append([1])
            coords.append([(1,1,1,1)])
            deltas = np.ones([p.GRID_SIZE*p.GRID_SIZE*p.ANCHOR_COUNT, 4])
            ret_deltas.append(deltas)

            iou_mask = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT])
            ret_gammas.append(iou_mask)

            input_mask = np.zeros([p.GRID_SIZE**2 * p.ANCHOR_COUNT])
            ret_masks.append(input_mask)

            classes = np.zeros([p.GRID_SIZE * p.GRID_SIZE * p.ANCHOR_COUNT,
                                p.OUT_CLASSES])
            ret_classes.append(classes)

            N_obj.append(1e0)'''

    return filename_list, N_obj, ret_deltas,\
        ret_gammas, ret_masks, ret_classes


def print_summary(image_data):
    """
    Debug method for checking read data about file.

    Takes an array defined as in the first lines of this method.

    Output to stdout.
    """
    name = image_data[0]
    labels = image_data[1]
    coords = image_data[2]
    deltas = image_data[3]
    gamma = image_data[4]
    mask = image_data[5]
    classes = image_data[6]
    flat_mask = np.reshape(mask, [-1, 1]).astype(int)
    print('Summary for file ' + name + ':')
    print('Labels in picture: ',)
    for label in labels:
        print(label)
    print('Coordinates for boxes in picture: ',)
    for c in coords:
        print(c,)
    print('Deltas calculated: ')
    for i, d in enumerate(deltas):
        if flat_mask[i] == 1:
            y = (i//9)//p.GRID_SIZE
            x = (i//9) % p.GRID_SIZE
            cl = np.argmax(classes[i])
            print('Delta for pos (%i,%i) to class %i with anchor %i, IOU %f: '
                  % (x, y, cl, i % 9, gamma[i]), end='')
            print(d)


def read_images_from_disk(filename, folder):
    """
    Consume a list of filenames, loads images from disc and transforms.

    Args:
      filename: An 1D string tensor.
    Returns:
      One tensor: the decoded images.
    """
    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.image.resize_images(image, [p.IMAGE_SIZE, p.IMAGE_SIZE])
    tf.image.convert_image_dtype(image,
                                 dtype=tf.float32,
                                 saturate=False,
                                 name=None)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.1)
    tf.image.convert_image_dtype(image,
                                 dtype=tf.uint8,
                                 saturate=False,
                                 name=None)
    return image


def get_batch(size, folder):
    """
    Master method for reading data.

    Input:
        size: batch size. Integer representing number of pictures to load.
        folder: path to folder from which to load data.

    Returns:
        Batched images, deltas, gammas masks, classes, object counts, each as
        Tensors of size [batch,[data]] where data is the size of each data type
    """
    image_list, Nobj_list, delta_list,\
        gamma_list, mask_list, class_list \
        = read_image_folders(folder)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    deltas = tf.convert_to_tensor(np.asarray(delta_list), dtype=tf.float32)
    gammas = tf.convert_to_tensor(np.asarray(gamma_list), dtype=tf.float32)
    masks = tf.convert_to_tensor(np.asarray(mask_list), dtype=tf.int32)
    classes = tf.convert_to_tensor(np.asarray(class_list), dtype=tf.int32)
    Nobj = tf.reshape(tf.convert_to_tensor(np.asarray(Nobj_list),
                                           dtype=tf.float32),
                      shape=[-1, 1])

    tensor_slice = tf.train.slice_input_producer(
        [images, deltas, gammas, masks, classes, Nobj], shuffle=True)

    image = read_images_from_disk(tensor_slice[0], folder)

    image_batch, delta_batch, gamma_batch,\
        mask_batch, class_batch, n_obj_batch\
        = tf.train.batch([image,
                         tensor_slice[1],  # deltas
                         tensor_slice[2],  # gammas
                         tensor_slice[3],  # masks
                         tensor_slice[4],  # classes
                         tensor_slice[5]],  # n_obj
                         batch_size=size)

    return image_batch, \
        delta_batch, gamma_batch, mask_batch, class_batch, n_obj_batch
