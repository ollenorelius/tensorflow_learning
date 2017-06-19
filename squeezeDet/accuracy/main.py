"""Script for evaluating accuracy in trained neural nets."""
from forward_net import NeuralNet
import sys
import os
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import utils as u
from matplotlib import pyplot as plt


def draw_boxes(boxes):
        mask = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
        d = ImageDraw.Draw(mask)
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf', 12)
        txt_offset_x = 0
        txt_offset_y = 20
        for box in boxes:
            p_coords = [box.coords[0]*(512, 512)[0],
                        box.coords[1]*(512, 512)[1],
                        box.coords[2]*(512, 512)[0],
                        box.coords[3]*(512, 512)[1]]
            if box.classification == 1:
                box_color = 'red'
            elif box.classification == 0:
                box_color = 'blue'
            else:
                box_color = 'green'
            d.rectangle(p_coords, outline=box_color)

            textpos = (p_coords[0] - txt_offset_x, p_coords[1] - txt_offset_y)
            d.text(textpos, 'Class %s at %s confidence' %
                   (box.classification, box.confidence),
                   font=fnt, fill=box_color)

        return mask


def get_ground_truths():
    """Collect ground truths from list.txt."""
    f = open('./data/test/list.txt')
    truth_boxes = {}
    truth_classes = {}
    for line in f:
        tokens = line.strip().split()
        name = tokens[0]
        box = [float(x) for x in tokens[1:5]]
        cl = int(tokens[5]) - 1
        ibox = u.inv_trans_boxes([box])
        ibox = np.squeeze(ibox)

        if name in truth_boxes.keys():
            truth_boxes[name].append(ibox)
        else:
            truth_boxes[name] = [ibox]

        if name in truth_classes.keys():
            truth_classes[name].append(cl)
        else:
            truth_classes[name] = [cl]
    return truth_boxes, truth_classes


name = sys.argv[1]
net = NeuralNet(name)

data_folder = './data/test'

filenames = []
filenames_unfiltered = os.listdir(data_folder)
tb, tc = get_ground_truths()

for unf in filenames_unfiltered:
    if re.search('\.jpg\Z', unf) is not None:
        filenames.append(data_folder + '/' + unf)


for i in range(10):
    recall = []
    precision = []
    cutoff = i/50 + 0.1
    print('cutoff: %s ' % cutoff, end="")
    '''batch_size = 10
    boxes = []

    i_file_slice = 0
    while i_file_slice < np.ceil(len(filenames)/batch_size):
        i_file_slice += 1
        slice_start = (i_file_slice-1)*batch_size
        slice_end = (i_file_slice)*batch_size
        if slice_end > len(filenames):
            slice_end = len(filenames)
        print(slice_end)
        file_slice = filenames[slice_start:slice_end]

        images = list(map(Image.open, file_slice))
        images = list(map(lambda x: x.resize((512, 512)), images))
        images = list(map(np.asarray, images))
        #print(images)
        batch_boxes = net.run_images(images, cutoff=cutoff)
        list(map(boxes.append, batch_boxes))
    box_dict = dict(zip(filenames, boxes))
    print(box_dict)'''
    for filename in filenames:
        image = Image.open(filename).resize((512, 512))
        filename = filename.split('/')[-1]
        if filename not in tb.keys():
            continue

        truth = np.asarray(tb[filename]).transpose()

        boxes = net.run_images([np.asarray(image)], cutoff=cutoff)
        TP = 0
        FP = 0
        FN = 0
        found = [0]*len(boxes[0])

        for i, box in enumerate(boxes[0]):
            t_box = np.squeeze(u.inv_trans_boxes([box.coords]))
            ious = u.intersection_over_union(bbox=t_box,
                                             anchors=truth)
            best_index = np.argmax(ious)
            if ious[best_index] > 0.5 \
                    and tc[filename][best_index] == box.classification:
                TP += 1
                found[i] += 1
            else:
                FP += 1
            for i in found:
                if i == 0:
                    FN += 1

        if TP + FP == 0:
            precision.append(0)
        else:
            precision.append(TP/(TP+FP))

        if TP + FN == 0:
            recall.append(0)
        else:
            recall.append(TP/(TP+FN))
        '''image = image.convert('RGBA')
        mask = draw_boxes(boxes[0])
        image = Image.alpha_composite(image, mask)
        image.show()
        input()'''
    f = 2*(np.mean(precision) * np.mean(recall)) \
           / (np.mean(precision) + np.mean(recall))
    print(f)
plt.scatter(recall, precision)
plt.show()
