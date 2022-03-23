# -*- coding:utf-8 -*-
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import io
import cv2
import time
import glob
import random
from scipy import ndimage
import numpy as np
from PIL import Image, ImageFilter
from precoss import *
from tqdm import tqdm

def get_list_from_filenames(file_path):
    with open(file_path,'r',) as f:
        lines = [one_line.strip('\n') for one_line in f.readlines()]
    return lines

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_annotation_dict(input_folder_path, word2number_dict):
    label_dict = {}
    for i in range(len(input_folder_path[:])):
        tmp = input_folder_path[i].split("/")[4]
        label_dict[input_folder_path[i]] = word2number_dict[tmp]
    return label_dict

def create_tf_example(image_path, label, resize=None):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encode_jpg = fid.read()
    encode_jpg_io = io.BytesIO(encode_jpg)
    image = Image.open(encode_jpg_io)
    if resize is not None:
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encoded_jpg = bytes_io.getvalue()
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image': bytes_feature(encode_jpg),
                'label': int64_feature(label),
            }
        ))
    return tf_example
   
def generate_tfrecord(annotation_dict, record_path, resize=None):
    num_tf_example = 0
    writer = tf.python_io.TFRecordWriter(record_path)
    for image_path, label in annotation_dict.items():
        if not tf.gfile.GFile(image_path):
            print("{} does not exist".format(image_path))
        tf_example = create_tf_example(image_path, label, resize)
        writer.write(tf_example.SerializeToString())
        num_tf_example += 1
        if num_tf_example % 100 == 0:
            print("Create %d TF_Example" % num_tf_example)
    writer.close()
    print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))

if __name__=="__main__":
    
    word2number_dict = {
        "0":0,
        "0_new":0,
        "0_new1":0,
        "0_new2":0,
        "0_new3": 0,
        "0_new4":0,
        "neg":0,
        "gen_minzui":0,
        "shoushi_extra": 0,
        "20220110_imgs_npy_0": 0,
        "ziya_kouhong_shoushi":0,
        "shoushi_quanzhedang":0,
        "1": 1,
        "1_train":1,
        "2":2,
        "2_new":2,
        "2_new_cp":2,
        "2_new_yisifuyangben": 2,
        "2_yisifuyangben": 2,
        "20220110_imgs_npy_2": 2,
        "ziya_shenshetou_2":2,
        "3":3,
        "3_train":3,
    }

    # word2number_dict = {
    #     "0":0,
    #     "1":1,
    #     "2":2,
    #     "3":3,
    # }

    MODE = "Train"
    record_path = "F:/data_segment/0316_train.record"
    images_list = get_list_from_filenames("f:/data_segment/0316_train_list4.txt")
    random.shuffle(images_list)
    print(len(images_list))
    annotation_dict = get_annotation_dict(images_list, word2number_dict)
    
    with tf.io.TFRecordWriter(record_path) as writer:
        for filename, label in tqdm(annotation_dict.items()):
            if filename[-3:]=="jpg":
                filename_format = "JPEG"
            elif filename[-3:]=="png":
                    filename_format = "PNG"
            else:
                print("filename_format error")
                
            with tf.gfile.GFile(filename, 'rb') as fid:
                encode_jpg = fid.read()
            encode_jpg_io = io.BytesIO(encode_jpg)
            image = Image.open(encode_jpg_io)
            
            if MODE == "Train":
                image, label = enforce_random(image, label, 0.4)
            image = np.asarray(image)
            if MODE == "Train" and (label==0 or label==2):
                image, label = motion_randon(image, label, 0.4)
                image, label = imgBrightness(image, label, 0.4)
            
            image = Image.fromarray(image)
            label = Image.fromarray(label)
            bytes_io = io.BytesIO()
            image.save(bytes_io, format=filename_format)
            encoded_jpg = bytes_io.getvalue()
            label.save(bytes_io, format=filename_format)
            encoded_label = bytes_io.getvalue()            
            
            feature = {                        
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),  
                'label': tf.train.Feature(int64_list=tf.train.BytesList(value=[encoded_label]))   
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString()) 
        writer.close()