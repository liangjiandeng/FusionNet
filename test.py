# This is the test file for the work:
# LJ. Deng, et al., "Detail Injection-based Deep Convolutional Neural Networks for Pansharpening", TGRS, 2020
# If you want to test your dataset, please add your .mat dataset to this folder with the format: NxHxWxC
# Also, we leave two test data, i.e., new_data6, for the test!
# Requirement: tf.1 version (based on PanNet's code)
# 2019-10-1


import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import scipy.io as sio
import time
import datetime
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def FusionNet(lms, pan, num_spectral = 8, num_res = 4, num_fm = 32, reuse=False):
    weight_decay = 1e-5
    with tf.variable_scope('net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()

        pan_concat = tf.concat([pan, pan, pan, pan, pan, pan, pan, pan], axis = 3)
        ms  = tf.subtract(pan_concat, lms)  # ms = pan - lms

        ##### ResNet #########################
        rs = ly.conv2d(ms, num_outputs=num_fm, kernel_size=3, stride=1,
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.relu)

        for i in range(num_res):
            rs1 = ly.conv2d(rs, num_outputs = num_fm, kernel_size = 3, stride = 1,
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = tf.nn.relu)

            rs1 = ly.conv2d(rs1, num_outputs = num_fm, kernel_size = 3, stride = 1,
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
            rs = tf.add(rs, rs1)
        
        rs = ly.conv2d(rs, num_outputs = num_spectral, kernel_size = 3, stride = 1,
                          weights_regularizer = ly.l2_regularizer(weight_decay), 
                          weights_initializer = ly.variance_scaling_initializer(),activation_fn = None)
        return rs

 ###########################################################################
 ###########################################################################
 ########### input data from here, (likes sub-funs in matlab before) ######

if __name__=='__main__':

    test_data = 'new_data6.mat'
    model_directory = './models/pre-trained-wv3'  # load the pre-trained model
    #model_directory = './models'  # load the trained model you get

    tf.reset_default_graph()
    
    data = sio.loadmat(test_data)  # load data

    ### input data for test! ######
    ms = data['ms'][...]      # MS image: 16x16x8
    ms = np.array(ms,dtype = np.float32) /2047.
    ms = ms[np.newaxis, :, :, :]    # convert to 4-D format (1x16x16x8): consistent with Net format!

    lms = data['lms'][...]    # up-sampled LRMS image: 64x164x8
    lms = np.array(lms, dtype = np.float32) /2047.
    lms = lms[np.newaxis, :, :, :]    # convert to 4-D format (1x64x64x8): consistent with Net format!
    
    pan = data['pan'][...]  # PAN image: 64x164x1
    pan = np.array(pan,dtype = np.float32) /2047.
    pan = pan[np.newaxis, :, :, np.newaxis]    # 4D format(1x64x64x1) a little different from before!

    h = pan.shape[1]  # height
    w = pan.shape[2]  # width
    
 ############## placeholder for tensor ################
    
    pan_test = tf.placeholder(shape=[1,h,w,1],dtype=tf.float32)
    ms_test = tf.placeholder(shape=[1,h/4,w/4,8],dtype=tf.float32)
    lms_test = tf.placeholder(shape=[1,h,w,8],dtype=tf.float32)

 ######## network architecture (call: PanNet constructed before!) #########
    
    rs = FusionNet(lms_test, pan_test)  # output high-frequency parts
    mrs = tf.add(rs, lms_test)
    output = tf.clip_by_value(mrs, 0, 1)  # final output

 ###########################################################################
 ###########################################################################
 #### Run the above (take real test_data into the Net, for test) ############
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:  
        sess.run(init)

        #### loading  model ######    
        if tf.train.get_checkpoint_state(model_directory):  # if there exists trained model, use it!
           ckpt = tf.train.latest_checkpoint(model_directory)
           saver.restore(sess, ckpt)
           print ("load new model")

        else:  # if there exists no trained model, use pre-trained model!
           ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
           saver.restore(sess,ckpt.model_checkpoint_path)
           print ("load pre-trained model")                            

        start = datetime.datetime.now()
        final_output = sess.run(output,feed_dict = {pan_test: pan, ms_test: ms, lms_test: lms})
        end = datetime.datetime.now()
        print('time cost=' + str(end-start) + 's')

        sio.savemat('./result/output_our_wv3_newdata6.mat', {'output_our_wv3_newdata6':final_output[0, :, :, :]})
