import tensorflow as tf
import params as p
import utils as u
import numpy as np
import os
import re



def convert_KITTI_list(folder):
    files = os.listdir(folder)
    pic_x_size = 1224
    pic_y_size = 370
    class_dict = {'Car':1, 'Van':2,
                    'Truck':3, 'Pedestrian':4,
                    'Person_sitting':5, 'Cyclist':6,
                    'Tram':7, 'Misc':8,
                    'DontCare':0}
    out_file = open(folder+'/list.txt', 'w')
    for f in files:
        if re.match('[0-9]{6}.txt', f):
            rf = open(folder+'/'+f, 'r')
            for line in rf:
                tokens = line.strip().split()
                x1_p = float(tokens[4])
                y1_p = float(tokens[5])
                x2_p = float(tokens[6])
                y2_p = float(tokens[7])

                conv_line = []
                conv_line.append(f.split('.')[0]+'.png')
                conv_line.append('{0:.3f}'.format(x1_p/pic_x_size))
                conv_line.append('{0:.3f}'.format(y1_p/pic_y_size))
                conv_line.append('{0:.3f}'.format(x2_p/pic_x_size))
                conv_line.append('{0:.3f}'.format(y2_p/pic_y_size))
                conv_line.append(class_dict[tokens[0]])
                conv_line.append('\n')
                conv_str = ' '.join([str(i) for i in conv_line])
                out_file.write(conv_str)
            rf.close()

    out_file.close()




def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing paths and labels
    Args:
       image_list_file: a .txt file with one /path/to/image
            followed by 4 coords [x1 y1 x2 y2],
            followed by a integer class per line.

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
        ret_classes =[]
        N_obj = []
    anchors = np.array(u.create_anchors(p.GRID_SIZE)) # KXY x 4
    for i in range(len(filenames)):
        N_obj.append(len(labels[i]))
        #which box is each anchor assigned to?
        box_mask = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT])

        #This is I_ijk in the paper
        input_mask = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT])

        #What is the IOU at every grid point and every anchor?
        iou_mask = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT])

        deltas = np.zeros([p.GRID_SIZE*p.GRID_SIZE*p.ANCHOR_COUNT, 4])

        it_coords = u.inv_trans_boxes(coords[i]) # x,y,w,h

        classes = np.zeros([p.GRID_SIZE* p.GRID_SIZE* p.ANCHOR_COUNT, p.OUT_CLASSES])

        ious = []

        for box in it_coords:
            ious.append(u.intersection_over_union(box, np.transpose(anchors)))
        ious = np.array(ious) #N_obj x XYK

        box_mask = np.argmax(ious, 0) # XYK x 1
        iou_mask = np.amax(ious, 0) # XYK x 1
        #print('box mask shape is: ', end='')
        #print(box_mask.shape)
        #print(np.reshape(iou_mask,[-1,9]))


        chosen_boxes = it_coords[box_mask,:]
        #print(filenames[i])
        #print(chosen_boxes)
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

        #Reshaping the IOUs to be a matrix of [grid points x anchors]
        iou_mask_per_grid_point = np.reshape(iou_mask, [p.GRID_SIZE**2, p.ANCHOR_COUNT])
        #Which anchor has the highest IOU at every grid point?
        input_mask_indices = np.argmax(iou_mask_per_grid_point, 1)

        input_mask = np.zeros([p.GRID_SIZE**2, p.ANCHOR_COUNT])
        for j in range(p.GRID_SIZE**2):
            if(iou_mask_per_grid_point[j,input_mask_indices[j]]) > 0.01:
                input_mask[j,input_mask_indices[j]] = 1
        #print(box_mask)
        for j in range(p.GRID_SIZE**2*p.ANCHOR_COUNT):
            classes[j,labels[i][box_mask[j]]] = 1

        ret_deltas.append(deltas)
        ret_gammas.append(iou_mask)
        ret_masks.append(input_mask)
        ret_classes.append(classes)

        '''print_summary((filenames[i],
                       labels[i],
                       coords[i],
                       ret_deltas[-1],
                       ret_gammas[-1],
                       ret_masks[-1],
                       ret_classes[-1] ))

        input()'''

    return filenames, N_obj, ret_deltas,\
           ret_gammas, ret_masks, ret_classes

def print_summary(image_data):
    name = image_data[0]
    labels = image_data[1]
    coords = image_data[2]
    deltas = image_data[3]
    gamma = image_data[4]
    mask = image_data[5]
    classes = image_data[6]
    flat_mask = np.reshape(mask, [-1,1]).astype(int)
    print('Summary for file ' + name + ':')
    print('Labels in picture: ',)
    for label in labels: print(label)
    print('Coordinates for boxes in picture: ',)
    for c in coords: print(c,)
    print('Deltas calculated: ')
    for i, d in enumerate(deltas):
        if flat_mask[i] == 1:
            y = (i//9)//p.GRID_SIZE
            x = (i//9)%p.GRID_SIZE
            cl = np.argmax(classes[i])
            print('Delta for pos (%i,%i) to class %i with anchor %i, IOU %f: '%(x,y,cl,i%9,gamma[i]),end='')
            print(d)
    print('Mask: ')
    for m in mask: print(m,)
    print('Classes: '+ '\n')
    for c in classes: print(c,)




def read_images_from_disk(filename,folder):
    """Consumes a single filename.
    Args:
      filename: A scalar string tensor.
    Returns:
      One tensor: the decoded image.
    """

    file_contents = tf.read_file(folder+'/training/image_2/'+filename)
    image = tf.image.decode_png(file_contents, channels=3)
    image = tf.image.resize_images(image, [256,256])
    return image

def get_batch(size,folder):
    image_list, Nobj_list, delta_list,\
      gamma_list, mask_list, class_list \
                    = read_labeled_image_list("%s/training/label_2/list.txt"%folder)

    #print(coord_list)
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    #coords = tf.convert_to_tensor(coord_list, dtype=tf.float32)
    deltas = tf.convert_to_tensor(np.array(delta_list), dtype=tf.float32)
    gammas = tf.convert_to_tensor(np.array(gamma_list), dtype=tf.float32)
    masks = tf.convert_to_tensor(np.array(mask_list), dtype=tf.int32)
    classes = tf.convert_to_tensor(np.array(class_list), dtype=tf.int32)
    Nobj = tf.reshape(tf.convert_to_tensor(np.array(Nobj_list), dtype=tf.float32), shape=[-1, 1])

    tensor_slice = tf.train.slice_input_producer(
        [images, deltas, gammas, masks, classes, Nobj], shuffle=True)

    image = read_images_from_disk(tensor_slice[0],folder)


    image_batch, delta_batch, gamma_batch,\
     mask_batch, class_batch, n_obj_batch\
                    = tf.train.batch([image,
                                    tensor_slice[1],
                                    tensor_slice[2],
                                    tensor_slice[3],
                                    tensor_slice[4],
                                    tensor_slice[5]],
                                    batch_size=size)

    return image_batch, \
    delta_batch, gamma_batch, mask_batch, class_batch, n_obj_batch
