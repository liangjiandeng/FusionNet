# This is the training file for the work:
# LJ. Deng, et al., "Detail Injection-based Deep Convolutional Neural Networks for Pansharpening", TGRS, 2020
# If you want to train your dataset, please add your .mat dataset to training_data folder with the format: NxHxWxC
# Requirement: tf.1 version (based on PanNet's code)
# 2019-10-1


import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.layers as ly
import os
import datetime
import scipy.io as sio
import h5py
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



 ########## visualization ####################
def vis_ms(data):
    _,b,g,_,r,_,_,_ = tf.split(data,8,axis = 3)
    vis = tf.concat([r,g,b],axis = 3)
    return vis

 ########## FusionNet structures ################
def FusionNet(lms, pan, num_spectral = 8, num_res = 4, num_fm = 32, reuse=False):
    
    weight_decay = 1e-5   ## default: 1e-5;
    with tf.variable_scope('net'):        
        if reuse:
            tf.get_variable_scope().reuse_variables()

        pan_concat = tf.concat([pan, pan, pan, pan, pan, pan, pan, pan], axis = 3)  # copy 8-band pan img --> pan_concat=32x64x64x8
        ms = tf.subtract(pan_concat, lms)   # ms_residual = pan - lms  (high-fre info);   ms = 32x64x64x8

        ####### ResNet part (following) ################default: kernel_size = 3, stride = 1,
        rs = ly.conv2d(ms, num_outputs=num_fm, kernel_size=3, stride=1,    # input layer of ResNet, de
                          weights_regularizer=ly.l2_regularizer(weight_decay),
                          weights_initializer=ly.variance_scaling_initializer(),
                          activation_fn = tf.nn.relu)

        for i in range(num_res):     # hidden layers of ResNet (here total = 2xnum_res = 8 layers)

            rs1 = ly.conv2d(rs, num_outputs=num_fm, kernel_size = 3, stride = 1,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu)

            rs1 = ly.conv2d(rs1, num_outputs = num_fm, kernel_size = 3, stride = 1,
                            weights_regularizer = ly.l2_regularizer(weight_decay),
                            weights_initializer = ly.variance_scaling_initializer(),
                            activation_fn = None)  # can use tf.nn.relu!

            rs = tf.add(rs,rs1)   # ResNet: identity part in ResNet

        rs = ly.conv2d(rs, num_outputs=num_spectral, kernel_size=3, stride=1,  # output(predict) layer of ResNet
                           weights_regularizer = ly.l2_regularizer(weight_decay),
                           weights_initializer = ly.variance_scaling_initializer(),
                           activation_fn = None) # can use tf.nn.relu!
        return rs

 ###########################################################################
 ###########################################################################
 ########### input data from here, (likes sub-funs in matlab before) ######
    
if __name__ =='__main__':

    tf.reset_default_graph()   

    train_batch_size = 32 # training batch size
    test_batch_size = 32  # validation batch size
    image_size = 64      # patch size
    iterations = 200100  # total number of iterations to use.
    model_directory = './models'  # directory to save trained model to.
    train_data_name = './training_data/train.mat'  # training data: make it in matlab in advance;
    test_data_name = './training_data/validation.mat'   # validation data: make it in matlab in advance;
    restore = False  # load model or not
    method = 'Adam'  # training method: Adam or SGD

    
 ############## loading data ##############
    train_data = sio.loadmat(train_data_name)  # case 1: for small data (only for training to tune code, not v7.3 data)
    test_data = sio.loadmat(test_data_name)

    #train_data = h5py.File(train_data_name)  # case 2: for large data (for real training v7.3 data in matlab)
    #test_data = h5py.File(test_data_name)

    ############## placeholder for training ###########
    gt = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,8])
    lms = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,8])
    ms = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size//4,image_size//4,8])
    pan = tf.placeholder(dtype = tf.float32,shape = [train_batch_size,image_size,image_size,1])
    

 ############# placeholder for testing ##############
    test_gt = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size,image_size,8])
    test_lms = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size,image_size,8])
    test_ms = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size//4,image_size//4,8])
    test_pan = tf.placeholder(dtype = tf.float32,shape = [test_batch_size,image_size,image_size,1])

 ######## network architecture (call: PanNet constructed before!) ######################
    mrs = FusionNet(lms, pan)
    mrs = tf.add(mrs, lms)     # last in the architecture: add two terms together
    
    test_rs = FusionNet(test_lms, test_pan, reuse = True)
    test_rs = test_rs + test_lms  # same as: test_rs = tf.add(test_rs,test_lms) 


 ######## loss function ################
    mse = tf.reduce_mean(tf.square(mrs - gt))  # compute cost
    test_mse = tf.reduce_mean(tf.square(test_rs - test_gt))

 ##### Loss summary (for observation) ################
    mse_loss_sum = tf.summary.scalar("mse_loss",mse)
    test_mse_sum = tf.summary.scalar("test_loss",test_mse)
    lms_sum = tf.summary.image("lms",tf.clip_by_value(vis_ms(lms),0,1))
    mrs_sum = tf.summary.image("rs",tf.clip_by_value(vis_ms(mrs),0,1))
    label_sum = tf.summary.image("label",tf.clip_by_value(vis_ms(gt),0,1))
    all_sum = tf.summary.merge([mse_loss_sum,mrs_sum,label_sum,lms_sum])
    
 ############ optimizer: Adam or SGD ##################
         
    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'net')    
    
    if method == 'Adam':          # default: (0.001, beta1 = 0.9)
        g_optim = tf.train.AdamOptimizer(0.0003, beta1 = 0.9) \
                          .minimize(mse, var_list=t_vars)

    else:
        global_steps = tf.Variable(0,trainable = False)
        lr = tf.train.exponential_decay(0.1,global_steps,decay_steps = 50000, decay_rate = 0.1)
        clip_value = 0.1/lr
        optim = tf.train.MomentumOptimizer(lr,0.9)
        gradient, var = zip(*optim.compute_gradients(mse,var_list = t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient,clip_value)
        g_optim = optim.apply_gradients(zip(gradient,var),global_step = global_steps)
        

 ###########################################################################
 ###########################################################################
 #### Run the above (take real data into the Net, for training) ############

    init = tf.global_variables_initializer()  # initialization: must done!

    saver = tf.train.Saver()
    with tf.Session() as sess:  
        sess.run(init)
 
        if restore:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess,ckpt.model_checkpoint_path)

        #### read training data #####
        gt1 = train_data['gt'][...]  ## ground truth N*H*W*C
        pan1 = train_data['pan'][...]  #### Pan image N*H*W
        ms_lr1 = train_data['ms'][...]  ### low resolution MS image
        lms1 = train_data['lms'][...]  #### MS image interpolation -to Pan scale

        gt1 = np.array(gt1, dtype=np.float32) / 2047.  ### normalization, WorldView L = 11
        pan1 = np.array(pan1, dtype=np.float32) / 2047.
        ms_lr1 = np.array(ms_lr1, dtype=np.float32) / 2047.
        lms1 = np.array(lms1, dtype=np.float32) / 2047.
        N = gt1.shape[0]

        #### read validation data #####
        gt2 = test_data['gt'][...]  ## ground truth N*H*W*C
        pan2 = test_data['pan'][...]  #### Pan image N*H*W
        ms_lr2 = test_data['ms'][...]  ### low resolution MS image
        lms2 = test_data['lms'][...]  #### MS image interpolation -to Pan scale

        gt2 = np.array(gt2, dtype=np.float32) / 2047.  ### normalization, WorldView L = 11
        pan2 = np.array(pan2, dtype=np.float32) / 2047.
        ms_lr2 = np.array(ms_lr2, dtype=np.float32) / 2047.
        lms2 = np.array(lms2, dtype=np.float32) / 2047.
        N2 = gt2.shape[0]

        mse_train = []
        mse_valid = []

        start = datetime.datetime.now()

        for i in range(iterations):  # totally 25500 iters
            ###################################################################
            #### training phase! ###########################

            bs = train_batch_size
            batch_index = np.random.randint(0, N, size=bs)

            train_gt = gt1[batch_index, :, :, :]
            train_pan = pan1[batch_index, :, :]
            train_pan = train_pan[:, :, :, np.newaxis]  # expand to N*H*W*1; new added!
            train_ms = ms_lr1[batch_index, :, :, :]
            train_lms = lms1[batch_index, :, :, :]

            #train_gt, train_lms, train_pan, train_ms = get_batch(train_data, bs = train_batch_size)
            _, mse_loss, merged = sess.run([g_optim,mse,all_sum],feed_dict = {gt: train_gt, lms: train_lms, ms: train_ms, pan: train_pan})


            if i % 100 == 0:

                print("Iter: " + str(i) + " MSE: " + str(mse_loss))   # print, e.g.,: Iter: 0 MSE: 0.18406609


            if i % 50000 == 0 and i != 0:  # save model each 50000 iters
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory + '/Iter_' + str(i) + '/model-' + str(i) + '.ckpt')
                print("Save Model")

            ###################################################################
            #### compute the mse of validation data ###########################
            bs_test = test_batch_size
            batch_index2 = np.random.randint(0, N2, size=bs_test)

            test_gt_batch = gt2[batch_index2, :, :, :]
            test_pan2 = pan2[batch_index2, :, :]
            test_pan_batch = test_pan2[:, :, :, np.newaxis]  # expand to N*H*W*1; new added!
            test_ms_batch = ms_lr2[batch_index2, :, :, :]
            test_lms_batch = lms2[batch_index2, :, :, :]

            test_mse_loss, merged = sess.run([test_mse, test_mse_sum],
                                             feed_dict={test_gt: test_gt_batch, test_lms: test_lms_batch,
                                                        test_ms: test_ms_batch, test_pan: test_pan_batch})


            if i%1000 == 0 and i!=0:
                print("Iter: " + str(i) + " Valid MSE: " + str(test_mse_loss))  # print, e.g.,: Iter: 0 MSE: 0.18406609

        end = datetime.datetime.now()
        print('time cost of Ours = ', str(end - start) + 's')






