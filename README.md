# FusionNet:
LJ. Deng, et al., "Detail Injection-based Deep Convolutional Neural Networks for Pansharpening", TGRS, 2020


This is the training-testing file for the work:
LJ. Deng, et al., "Detail Injection-based Deep Convolutional Neural Networks for Pansharpening", TGRS, 2020
If you want to train your dataset, please add your .mat dataset to training_data folder with the format: # NxHxWxC
2019-10-1

Requirement: tf.1 version (based on PanNet's code)

#########################################################
#########################################################

# HOW TO RUN:

1) for training:

YOU shoud simulate your training (validation) dataset in matlab in advance, 

then save them as as -v7.3 **.mat format  or **.h5 file (by CxWxHxN format),

finally read them in python by: h5py.File function (see line 87-88 in the train.py)


The images for training (validation, testing) dataset can be downloaded from the website:

http://www.digitalglobe.com/samples?search=Imagery



2) for testing:

Also simulate your test dataset in matlab in advance, 

you may save them as **.mat format(NxHxWxC format, because they are small size),

finally read them in python by: sio.loadmat function (see line 61 in the test.py)


Note that, we have put all trained models for WV3, QB, Gaofen in the folder of "model". 

You may use the trained model for comparisons directly!


In the test, we only leave a WV3 test data, i.e., new_data6.mat, for the testing, you can change your

test data (also need to change the pre-trained model accordingly).



#########################################################
#########################################################

# Another two issues:

1) We only provide the code for WV3 dataset, you may try more datasets, e.g., QB, Gaofen, by above way.

2) We will release the pytorch version ASAP!


Any questions, let me know! 

Liangjian Deng (UESTC, liangjian.deng@uestc.edu.cn)

# Only use for education/research purpose
