#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import os
from tqdm import tqdm
from glob import glob
import shutil

def split(root_dir, save_dir):
    files = glob(os.path.join(root_dir, '*.jpg'))
    files.sort(key=lambda x:x)

    bnd = int(len(files)/10)
    index = 0

    for i,file in tqdm(enumerate(files)):
        i += 1
        print(file.split('/'))
        save_sub_dir = os.path.join(save_dir, file.split('\\')[0].split('/')[-1]+'part_%d'%(index))
        print(save_sub_dir)
        if not os.path.exists(save_sub_dir):
            os.makedirs(save_sub_dir)
        shutil.copy(file, save_sub_dir)

        if i%bnd==0:
            index += 1


root_dir = 'data/post_ori/test/play/v006 play_Trim1'
save_dir = 'data/post/test/play/'

split(root_dir, save_dir)




