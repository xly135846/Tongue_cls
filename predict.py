import tensorflow as tf

import os
import cv2
import glob
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix

from model import mobilenet

os.environ["CUDA_VISIBLE_DEVICES"]="7"

def get_list_from_filenames(file_path):
    with open(file_path,'r',) as f:
        lines = [one_line.strip('\n') for one_line in f.readlines()]
    return lines

def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    image = feature_dict['image']
    label = feature_dict['label']
    image = (tf.cast(image, dtype='float32')-127.5)/127.5
    return image, label

def predict(val_dataset, model,is_save):
    all_true,all_pred = [],[]
    count = 0
    for batch_idx, (images, labels) in enumerate(tqdm(val_dataset)):
        outputs = model.predict(images)
        all_pred += list(np.argmax(outputs, axis=1))
        all_true += list(labels)
        # all_true += list(np.argmax(labels, axis=1))
        pred_list = list(np.argmax(outputs, axis=1))
        if is_save==True:
            for i in range(len(labels)):
                if labels[i]==2 and pred_list[i]==0:
                    count += 1 
                    name = "./error/"+str(count)+"_"+str(labels[i].numpy())
                    for j in range(4):
                        name += "_"+str(outputs[i][j])
                    name += ".png"
                    cv2.imwrite(name, np.array(images[i,:,:,0]*127.5+127.5))
    cm = confusion_matrix(all_true, all_pred)
    print(cm)
    print("----Recall----")
    for i in range(len(cm)):
        print(i, cm[i][i]/sum(cm[i]))
    print("----Accuracy----")
    for i in range(len(cm)):
        print(i, cm[i][i]/sum(cm[:,i]))

    # loss, acc = model.evaluate(val_dataset)
    # print('evaluate',loss,acc)


if __name__=="__main__":

    batch_size = 512
    val_tfrecord_path = "/mnt/fu07/xueluoyang/data/val_class.record"

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    raw_dataset = tf.data.TFRecordDataset(val_tfrecord_path, num_parallel_reads=4)
    val_dataset = raw_dataset.map(_parse_example)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_batch = val_dataset.batch(batch_size)

    model = mobilenet(channels=[12,12,12,12,24,24,24,24,36,36,36,36,48,48,48,128], t=1)
    model.build(input_shape=(batch_size,100,100,1))

    all_weights = glob.glob("./checkpoints/tmp_log/*")
    all_weights.sort()
    # all_weights = ["./checkpoints/0314_log/cp_0260.hdf5",
    #                "./checkpoints/0314_log/cp_0353.hdf5",
    #                "./checkpoints/0314_log/cp_0366.hdf5",
    #                "./checkpoints/0314_log/cp_0371.hdf5",
    #                ]
    ### dataloader_test
    val_dataset = val_dataset.batch(batch_size)
    for i in range(len(all_weights[:])):
        print(all_weights[i])
        model.load_weights(all_weights[i])
        predict(val_dataset,model,False)