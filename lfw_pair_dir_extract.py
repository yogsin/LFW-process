#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
lfw_pair_dir_extract.py reads the lfw pair.txt file to extract image pair and 
save the two subimage imformation into two different files,
each file contains the image path and a label.
if the two labels in the same line from coresponding file are different, 
it means that the two images are from two dfferent identities
"""
from PIL import Image
import sys
import os
import random
import numpy as np


def set_to_csv_file(img_label_lib, file_name):
    f = open(file_name, 'wb')
    for item in img_label_lib:
        line = item[0] + ' ' + str(item[1]) + '\n'
        f.write(line)
    f.close()

def read_csv_file(csv_file):
    path_and_labels = []
    f = open(csv_file, 'rb')
    for line in f:
        line = line.strip('\r\n')
        path, label = line.split(' ')
        label = int(label)
        path_and_labels.append((path, label))
    f.close()
    random.shuffle(path_and_labels)
    return path_and_labels

def read_lfw_pair(lfw_pair_list = '/home/work/Jimmy/caffe-face/face_example/data/pairs.txt', 
#    lfw_dir  = 'E:/chen/paper/face/data/public_database/LFW/lfw/lfw/', 
    lfw_l_file = '/home/work/Jimmy/caffe-face/face_example/data/lfw_l_info', 
    lfw_r_file = '/home/work/Jimmy/caffe-face/face_example/data/lfw_r_info', 
    lfw_aligned_dir = '/home/work/DataSource/lfw_aligned/'):
#    if not lfw_dir.endswith('/'):
#        lfw_dir += '/'
    f = open(lfw_pair_list, 'rb')
    cur_line = 0    #当前行号
    lfw_l = []    #图像对子图1 匹配标志(0)
    lfw_r = []    #图像对子图2 匹配标志(0->同一个人，1->不同的人)  匹配标志作为标签信息，用于辅助计算准确率
    lfw_l_path = ''
    lfw_r_path = ''
    for line in f:
        print cur_line
        #忽略首行
        if cur_line == 0:
            cur_line += 1
            continue
        #来自同一个人
        if (cur_line-1) % 600 <300:
            line = line.strip('\r\n')
            path, index_l, index_r = line.split('\t')
            index_l = index_l.zfill(4)
            index_r = index_r.zfill(4)
            #子图1路径保存
            lfw_l_path = path +'/' + path + '_' + index_l + '.jpg'
            #子图2路径保存
            lfw_r_path = path +'/' + path + '_' + index_r + '.jpg'
            if os.path.exists(lfw_aligned_dir + lfw_l_path) and os.path.exists(lfw_aligned_dir + lfw_r_path):
                lfw_l.append((lfw_aligned_dir + lfw_l_path, 0))
                lfw_r.append((lfw_aligned_dir + lfw_r_path, 0))
            else:
                print lfw_aligned_dir + lfw_l_path,lfw_aligned_dir + lfw_r_path
        #来自不同的人
        elif (cur_line-1) % 600 <600:
            line = line.strip('\r\n')
            path_l, index_l, path_r, index_r = line.split('\t')
            index_l = index_l.zfill(4)
            index_r = index_r.zfill(4)
            #子图1路径保存
            lfw_l_path = path_l +'/' + path_l + '_' + index_l + '.jpg'
            #子图2路径保存
            lfw_r_path = path_r +'/' + path_r + '_' + index_r + '.jpg'
            if os.path.exists(lfw_aligned_dir + lfw_l_path) and os.path.exists(lfw_aligned_dir + lfw_r_path):
                lfw_l.append((lfw_aligned_dir + lfw_l_path, 0))
                lfw_r.append((lfw_aligned_dir + lfw_r_path, 1))
            else:
                print lfw_aligned_dir + lfw_l_path,lfw_aligned_dir + lfw_r_path
        cur_line +=1
    f.close()
    #将子图路径与标签信息保存到文档中
    set_to_csv_file(lfw_l, lfw_l_file)
    set_to_csv_file(lfw_r, lfw_r_file)
#    random.shuffle(path_and_labels)
        
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: python %s lfw_pair_list, lfw_l_file, lfw_r_file, lfw_aligned_dir' % (sys.argv[0])
        sys.exit()
    lfw_pair_list = sys.argv[1]
#    lfw_dir = sys.argv[2]
    lfw_l_file = sys.argv[2]
    lfw_r_file = sys.argv[3]
    lfw_aligned_dir = sys.argv[4]
#    if not lfw_dir.endswith('/'):
#        lfw_dir += '/'
    if not lfw_aligned_dir.endswith('/'):
        lfw_aligned_dir += '/'
    read_lfw_pair(lfw_pair_list, lfw_l_file, lfw_r_file, lfw_aligned_dir)
    
