import tensorflow as tf
import params as p
import utils as u
import numpy as np


def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file,
                                        list of list of labels in each pic,
                                        list of list of coord tuples in each pic
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    coords = []

    for line in f:
        data = line.split(' ')

        if data[0] in filenames:
            ind = filenames.index(data[0])
            labels[ind].append(int(data[5]))
            coords[ind].append( (float(data[1]),
                            float(data[2]),
                            float(data[3]),
                            float(data[4])) )
        else:
            filenames.append(data[0])
            labels.append([int(data[5])])
            coords.append( [(float(data[1]),
                            float(data[2]),
                            float(data[3]),
                            float(data[4]))] )


        ret_deltas = []
        ret_gammas = []
        ret_masks  = []

    for i in range(len(filenames)):
        #which box is each anchor assigned to?
        box_mask = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT])

        #This is I_ijk in the paper
        input_mask = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT])

        #What is the IOU at every grid point and every anchor?
        iou_mask = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT])

        deltas = np.zeros([p.GRID_SIZE*p.GRID_SIZE*p.ANCHOR_COUNT, 4])
        anchors = np.array(u.create_anchors(p.GRID_SIZE)) # KXY x 4
        it_coords = u.inv_trans_boxes(coords[i]) # x,y,w,h

        classes = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT, p.OUT_CLASSES])

        ious = []

        for box in coords[i]:
            ious.append(u.intersection_over_union(np.transpose(box), np.transpose(anchors)))


        box_mask = np.argmax(ious, 0)
        iou_mask = np.amax(ious, 0)


        chosen_boxes = it_coords[box_mask,:]
        xg = chosen_boxes[:,0]
        yg = chosen_boxes[:,1]
        wg = chosen_boxes[:,2]
        hg = chosen_boxes[:,3]

        x_hat = anchors[:,0]
        y_hat = anchors[:,1]
        w_hat = anchors[:,2]
        h_hat = anchors[:,3]
        #print(wg)
        deltas[:,0] = (xg-x_hat)/w_hat
        deltas[:,1] = (yg-y_hat)/h_hat
        deltas[:,2] = np.log(wg/w_hat)
        deltas[:,3] = np.log(hg/h_hat)

        iou_mask_per_grid_point = np.reshape(iou_mask, [p.GRID_SIZE**2, p.ANCHOR_COUNT])
        input_mask_indices = np.argmax(iou_mask_per_grid_point, 0)
        input_mask = np.zeros([p.GRID_SIZE**2, p.ANCHOR_COUNT])
        for j in range(p.GRID_SIZE**2):
            input_mask[i,input_mask_indices[i]] = 1

        for j in range(p.GRID_SIZE**2*p.ANCHOR_COUNT)
            classes[i,classes[box_mask[i]]] = 1

        ret_deltas.append(deltas)
        ret_gammas.append(iou_mask)
        ret_masks.append(input_mask)





    return filenames, labels, coords, ret_deltas, ret_gammas, ret_masks, classes

def read_images_from_disk(filename):
    """Consumes a single filename.
    Args:
      filename: A scalar string tensor.
    Returns:
      One tensor: the decoded image.
    """

    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.image.resize_images(image, [256,256])
    return image

def get_batch(size,folder):
    image_list, label_list, coord_list, delta_list, gamma_list, mask_list \
                    = read_labeled_image_list("%s/list.txt"%folder)
    print(np.shape(gamma_list))
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    coords = tf.convert_to_tensor(coord_list, dtype=tf.float32)
    deltas = tf.convert_to_tensor(np.array(delta_list), dtype=tf.float32)
    gammas = tf.convert_to_tensor(np.array(gamma_list), dtype=tf.float32)
    masks = tf.convert_to_tensor(np.array(mask_list), dtype=tf.int32)
    tf.reshape(labels,[-1])

    tensor_slice = tf.train.slice_input_producer(
        [images, labels, coords, deltas, gammas, masks], shuffle=False)

    image = read_images_from_disk(tensor_slice[0])


    image_batch, label_batch, coord_batch, delta_batch, gamma_batch, mask_batch\
                    = tf.train.batch([image,
                                    tensor_slice[1],
                                    tensor_slice[2],
                                    tensor_slice[3],
                                    tensor_slice[4],
                                    tensor_slice[5]],
                                    batch_size=size)
    return image_batch, label_batch, coord_batch, delta_batch, gamma_batch, mask_batch
