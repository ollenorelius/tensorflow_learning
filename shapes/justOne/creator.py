import params
import numpy as np
import math
import cairo
import random
import scipy.misc
import os


def create_circle():
    """
        Create a blue circle of varying size in a black numpy array.

        Out: size^2 x channels numpy array
    """
    x = random.random()
    y = random.random()
    radius = random.random() * params.CIRCLE_VAR +  params.CIRCLE_RADIUS
    image = np.zeros(( params.IMAGE_SIZE, params.IMAGE_SIZE,4), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(
        image, cairo.FORMAT_ARGB32,  params.IMAGE_SIZE,  params.IMAGE_SIZE)
    cr = cairo.Context(surface)



    cr.arc(x * params.IMAGE_SIZE, y * params.IMAGE_SIZE, radius, 0, 2*math.pi)
    cr.set_line_width(5)
    cr.set_source_rgb(1.0, 0.0, 0.0)
    cr.stroke()

    return ((x,y), image)

def create_rect():
    """
        Create a blue square in a black numpy array.

        Out: size^2 x channels numpy array
    """
    x = random.random()
    y = random.random()
    radius = random.random() *  params.SQUARE_VAR +  params.SQUARE_SIZE
    image = np.zeros(( params.IMAGE_SIZE, params.IMAGE_SIZE,4), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(
        image, cairo.FORMAT_ARGB32,  params.IMAGE_SIZE,  params.IMAGE_SIZE)
    cr = cairo.Context(surface)



    cr.rectangle(x * params.IMAGE_SIZE, y * params.IMAGE_SIZE, params.SQUARE_SIZE, params.SQUARE_SIZE)
    cr.set_line_width(5)
    cr.set_source_rgb(1.0, 0.0, 0.0)
    cr.stroke()

    return ((x,y), image)

def create_justOne_batch(count):

    """
        Creates a stack of images together with their classes and coordinates.

        Coordinates are relative (eg [0, 1]) and start in top left corner.

        In: size of the batch (number of pictures)

        Out: tuple of:
            images: (count x imagesize x imagesize x channels) numpy array
            classes: (count x classes) one-hot array of labels
            coords: list of tuples: (x,y) x count


    """
    out = np.zeros([count, params.IMAGE_SIZE, params.IMAGE_SIZE, 4])
    classes = np.zeros([count,3])
    coords = np.zeros([count,2])
    for i in range(count):
        r = random.random()
        if r > 0.6:
            (sample_coords, data) = create_rect()
            out[i,:,:,:] = data
            coords[i,:] = sample_coords
            classes[i,2] = 1
        elif r > 0.3:
            (sample_coords, data) = create_circle()
            out[i,:,:,:] = data
            coords[i,:] = sample_coords
            classes[i,1] = 1
        #if r < 0.3 put nothing, class = 0, coords = (0,0)
        else:
            classes[i,0] = 1
    return (out, classes, coords)


def save_justOne_batch_to_disk(folder, size):
    """
        Creates a batch of <size> images and saves it to disk in <folder>.
        Images are numbered from 0.jpg to <size>.jpg
        Additionally saves labeling in file called list.txt in the same folder.

        label file is formatted as <file name> <class> <x> <y>
    """
    data = create_justOne_batch(size)

    if not os.path.exists(folder):
        os.makedirs(folder)

    strings = []
    for i in range(size):

        """print("Folder: %s" % folder)
        print("index: %i" % i)
        print("x: %f" % data[2][i][0])
        print("y: %f" % data[2][i][1])
        print("class:\n" % np.argmax(data[1][i]))"""
        scipy.misc.toimage(data[0][i,:,:,:], cmin=0.0, cmax=255).save('%s/%i.jpg'%(folder, i))
        strings.append('%s/%d.jpg %i %f %f\n'%(folder, i, np.argmax(data[1][i]), data[2][i][0], data[2][i][1]))

        list_file = open("%s/list.txt"%folder, 'w')
        for line in strings:
            list_file.write(line)
    return 1
