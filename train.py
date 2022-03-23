import tensorflow as tf
import keras.backend as K

import os
import gc
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

from model import mobilenet
from process import *

random.seed(2021)

os.environ["CUDA_VISIBLE_DEVICES"]="5"

def amsoftmax_loss(y_true, y_pred, scale=30, margin=1.0):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    image = feature_dict['image']
    label = feature_dict['label']
    image = (tf.cast(image, dtype='float32')-127.5)/127.5
    return image, label

def data_generator(dataset, batch_size):
    images = []
    labels = []
    for image,label in train_dataset.take(batch_size):
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    
    yield (images, labels)

def scheduler(epoch):
    if epoch < 5:
        return 0.001
    else:
        lr = 0.001 * tf.math.exp(0.1 * (5 - epoch))
        return lr.numpy()

if __name__=="__main__":

    batch_size = 512
    num_epochs = 500
    initial_learning_rate = 0.0005
    # trian_tfrecord_path = "/mnt/fu07/xueluoyang/data/aaa.record"
    trian_tfrecord_list = glob.glob("/mnt/fu07/xueluoyang/data/0316/*.record")
    val_tfrecord_path = "/mnt/fu07/xueluoyang/data/val_class.record"
    checkpoint_path = "./checkpoints/tmp_log/cp_{epoch:04d}.hdf5"
    LOG_DIR = "./checkpoints/tmp_fitlogs/"

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    # for kkk in range(int(num_epochs)):
    # tf.keras.backend.clear_session()
    
    raw_dataset = tf.data.TFRecordDataset(trian_tfrecord_list, num_parallel_reads=4)
    train_dataset = raw_dataset.map(_parse_example)
    train_dataset = train_dataset.shuffle(200000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
    train_batch = train_dataset.batch(batch_size)

    raw_dataset = tf.data.TFRecordDataset(val_tfrecord_path, num_parallel_reads=4)
    val_dataset = raw_dataset.map(_parse_example)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_batch = val_dataset.batch(batch_size)

    model = mobilenet(channels=[12,12,12,12,24,24,24,24,36,36,36,36,48,48,48,128], t=1)
    # if os.path.exists("./checkpoints/0310_log/cp_"+str(kkk-1)+".hdf5")==True:
    # model.load_weights("./checkpoints/0314_log/cp_0416.hdf5")

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
    #                                                              decay_steps=15,
    #                                                              decay_rate=0.95)

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate,),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                # loss=amsoftmax_loss,
                metrics=['accuracy'])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    HISTORY = model.fit(train_batch,
                        epochs=num_epochs,
                        steps_per_epoch=2087,
                        validation_data=val_batch,
                        validation_steps=21,
                        callbacks=[checkpointer, tensorboard_callback],
                        shuffle=True,
                        )

    # tf.keras.backend.clear_session()
    # gc.collect()