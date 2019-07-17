import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join



classes = ["Pedestrian", "Car", "Cyclist"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('Annotations/%s.xml'%(image_id))
    out_file = open('labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult)==1:
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()


if not os.path.exists('labels/'):
    os.makedirs('labels/')
# image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
cur_dir=os.getcwd()
labels_dir=os.path.join(cur_dir,'Annotations')
list_file = open('train.txt', 'w')
for parent, dirnames, filenames in os.walk(labels_dir):
    for file_name in filenames:
        name= file_name[:-4]
        list_file.write('%s/image_2/%s.jpg\n'%(wd, name))
        convert_annotation(name)
list_file.close()

# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

