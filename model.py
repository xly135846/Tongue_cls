import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import layers

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="4"

def mobilenet(channels=[12,12,12,12,24,24,24,24,36,36,36,36,
                        48,48,48,128], t=1):
    inputs = Input((100, 100, 1))
    conv_1 = Conv2D(channels[0], 
                    (3,3),
                    strides=(2,2),
                    padding='same',
                    #activation=tf.nn.relu6,
                    kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                    name='conv_1')(inputs)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_1_ac = Activation(tf.nn.relu6)(conv_1_bn)
    # print("111",conv_1_ac.shape)

    block_1_conv_1 = Conv2D(channels[1]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_1_conv_1')(conv_1_ac)
    block_1_conv_1_bn = BatchNormalization()(block_1_conv_1)
    block_1_conv_1_ac = Activation(tf.nn.relu6)(block_1_conv_1_bn)
    # print("222",block_1_conv_1_ac.shape)
    block_1_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',
                                        #activation=tf.nn.relu6,
                                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_1_deconv_1')(block_1_conv_1_ac)
    block_1_deconv_1_bn = BatchNormalization()(block_1_deconv_1)
    block_1_deconv_1_ac = Activation(tf.nn.relu6)(block_1_deconv_1_bn)
    # print("333",block_1_deconv_1_ac.shape)
    block_1_conv_2 = Conv2D(channels[2],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_1_conv_2')(block_1_deconv_1_ac)
    block_1_conv_2_bn = BatchNormalization()(block_1_conv_2)
    block_1_conv_2_ac = Activation(tf.nn.relu6)(block_1_conv_2_bn)
    # print("444",block_1_conv_2_ac.shape)
    block_1_add = conv_1_ac + block_1_conv_2_ac
    # print("555",block_1_add.shape)

    block_2_conv_1 = Conv2D(channels[3]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_2_conv_1')(block_1_add)
    block_2_conv_1_bn = BatchNormalization()(block_2_conv_1)
    block_2_conv_1_ac = Activation(tf.nn.relu6)(block_2_conv_1_bn)
    # print("666",block_2_conv_1_ac.shape)
    block_2_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(2,2),
                                        depth_multiplier=1,
                                        padding='same',
                                        #activation=tf.nn.relu6,
                                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_2_deconv_1')(block_2_conv_1_ac)
    block_2_deconv_1_bn = BatchNormalization()(block_2_deconv_1)
    block_2_deconv_1_ac = Activation(tf.nn.relu6)(block_2_deconv_1_bn)
    # print("777",block_2_deconv_1_ac.shape)
    block_2_conv_2 = Conv2D(channels[4],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_2_conv_2')(block_2_deconv_1_ac)
    block_2_conv_2_bn = BatchNormalization()(block_2_conv_2)
    block_2_conv_2_ac = Activation(tf.nn.relu6)(block_2_conv_2_bn)
    # print("888",block_2_conv_2_ac.shape)

    block_3_conv_1 = Conv2D(channels[5]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_3_conv_1')(block_2_conv_2_ac)
    block_3_conv_1_bn = BatchNormalization()(block_3_conv_1)
    block_3_conv_1_ac = Activation(tf.nn.relu6)(block_3_conv_1_bn)
    # print("999",block_3_conv_1_ac.shape)
    block_3_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',
                                        #activation=tf.nn.relu6,
                                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_3_deconv_1')(block_3_conv_1_ac)
    block_3_deconv_1_bn = BatchNormalization()(block_3_deconv_1)
    block_3_deconv_1_ac = Activation(tf.nn.relu6)(block_3_deconv_1_bn)
    # print("111",block_3_deconv_1_ac.shape)
    block_3_conv_2 = Conv2D(channels[6],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_3_conv_2')(block_3_deconv_1_ac)
    block_3_conv_2_bn = BatchNormalization()(block_3_conv_2)
    block_3_conv_2_ac = Activation(tf.nn.relu6)(block_3_conv_2_bn)
    # print("222",block_3_conv_2_ac.shape)
    block_3_add = block_2_conv_2_ac+block_3_conv_2_ac

    block_4_conv_1 = Conv2D(channels[7]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_4_conv_1')(block_3_add)
    block_4_conv_1_bn = BatchNormalization()(block_4_conv_1)
    block_4_conv_1_ac = Activation(tf.nn.relu6)(block_4_conv_1_bn)
    # print("333",block_4_conv_1_ac.shape)
    block_4_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(2,2),
                                        depth_multiplier=1,
                                        padding='same',
                                        #activation=tf.nn.relu6,
                                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_4_deconv_1')(block_4_conv_1_ac)
    block_4_deconv_1_bn = BatchNormalization()(block_4_deconv_1)
    block_4_deconv_1_ac = Activation(tf.nn.relu6)(block_4_deconv_1_bn)
    # print("444",block_4_deconv_1_ac.shape)
    block_4_conv_2 = Conv2D(channels[8],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_4_conv_2')(block_4_deconv_1_ac)
    block_4_conv_2_bn = BatchNormalization()(block_4_conv_2)
    block_4_conv_2_ac = Activation(tf.nn.relu6)(block_4_conv_2_bn)
    # print("555",block_4_conv_2_ac.shape)

    block_5_conv_1 = Conv2D(channels[9]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_5_conv_1')(block_4_conv_2_ac)
    block_5_conv_1_bn = BatchNormalization()(block_5_conv_1)
    block_5_conv_1_ac = Activation(tf.nn.relu6)(block_5_conv_1_bn)
    # print("666",block_5_conv_1_ac.shape)
    block_5_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',
                                        #activation=tf.nn.relu6,
                                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_5_deconv_1')(block_5_conv_1_ac)
    block_5_deconv_1_bn = BatchNormalization()(block_5_deconv_1)
    block_5_deconv_1_ac = Activation(tf.nn.relu6)(block_5_deconv_1_bn)
    # print("777",block_5_deconv_1_ac.shape)
    block_5_conv_2 = Conv2D(channels[10],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_5_conv_2')(block_5_deconv_1_ac)
    block_5_conv_2_bn = BatchNormalization()(block_5_conv_2)
    block_5_conv_2_ac = Activation(tf.nn.relu6)(block_5_conv_2_bn)
    # print("888",block_5_conv_2_ac.shape)
    block_5_add = block_4_conv_2_ac+block_5_conv_2_ac

    block_6_conv_1 = Conv2D(channels[11]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_6_conv_1')(block_5_add)
    block_6_conv_1_bn = BatchNormalization()(block_6_conv_1)
    block_6_conv_1_ac = Activation(tf.nn.relu6)(block_6_conv_1_bn)
    # print("999",block_6_conv_1_ac.shape)
    block_6_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(2,2),
                                        depth_multiplier=1,
                                        padding='same',
                                        #activation=tf.nn.relu6,
                                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_6_deconv_1')(block_6_conv_1_ac)
    block_6_deconv_1_bn = BatchNormalization()(block_6_deconv_1)
    block_6_deconv_1_ac = Activation(tf.nn.relu6)(block_6_deconv_1_bn)
    # print("111",block_6_deconv_1_ac.shape)
    block_6_conv_2 = Conv2D(channels[12],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_6_conv_2')(block_6_deconv_1_ac)
    block_6_conv_2_bn = BatchNormalization()(block_6_conv_2)
    block_6_conv_2_ac = Activation(tf.nn.relu6)(block_6_conv_2_bn)
    # print("222",block_6_conv_2_ac.shape)

    block_7_conv_1 = Conv2D(channels[13]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_7_conv_1')(block_6_conv_2_ac)
    block_7_conv_1_bn = BatchNormalization()(block_7_conv_1)
    block_7_conv_1_ac = Activation(tf.nn.relu6)(block_7_conv_1_bn)
    # print("333",block_7_conv_1_ac.shape)
    block_7_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',
                                        #activation=tf.nn.relu6,
                                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_7_deconv_1')(block_7_conv_1_ac)
    block_7_deconv_1_bn = BatchNormalization()(block_7_deconv_1)
    block_7_deconv_1_ac = Activation(tf.nn.relu6)(block_7_deconv_1_bn)
    # print("444",block_7_deconv_1_ac.shape)
    block_7_conv_2 = Conv2D(channels[14],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_7_conv_2')(block_7_deconv_1_ac)
    block_7_conv_2_bn = BatchNormalization()(block_7_conv_2)
    block_7_conv_2_ac = Activation(tf.nn.relu6)(block_7_conv_2_bn)
    # print("555",block_7_conv_2_ac.shape)
    block_7_add = block_6_conv_2_ac+block_7_conv_2_ac

    block_8_conv_1 = Conv2D(channels[15], 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_8_conv_1')(block_7_add)
    block_8_conv_1_ac = Activation(tf.nn.relu6)(block_8_conv_1)
    block_8_avg_pool = AveragePooling2D((7,7))(block_8_conv_1_ac)
    block_8_conv_2 = Conv2D(4, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #activation=tf.nn.relu6,
                            name='block_8_conv_2')(block_8_avg_pool)
    block_8_detect = tf.squeeze(block_8_conv_2, [1, 2])
    block_8_softmax = tf.nn.softmax(block_8_detect, name= 'softmax')
    
    model = Model(inputs=inputs, outputs=block_8_softmax)

    return model

if __name__=="__main__":

    model = mobilenet(channels=[12,12,12,12,24,24,24,24,36,36,36,36,
                        48,48,48,128],t=2)
    model.summary()

    print("success")