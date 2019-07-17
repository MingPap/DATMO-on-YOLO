import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np


wd = getcwd()


if not os.path.exists('set/'):
    os.makedirs('set/')

cur_dir=os.getcwd()
train_dir=os.path.join(cur_dir,'training')
test_dir=os.path.join(cur_dir,'testing')

list_file = open('set/train.txt', 'w')
a = np.uint16(len(os.listdir(train_dir))/2)
for i in range(a):
        list_file.write('%s/training/%06d.png\n'%(wd, i))
list_file.close()

list_file1 = open('set/test.txt', 'w')
b = np.uint16(len(os.listdir(test_dir))/2)
for i in range(b):
        list_file1.write('%s/testing/%06d.png\n'%(wd, i+a))
list_file1.close()

# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

