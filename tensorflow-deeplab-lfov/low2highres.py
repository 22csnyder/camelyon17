import argparse
from datetime import datetime
import os
import sys
import time
from skimage import io
import multiresolutionimageinterface as mir
from skimage.filters import threshold_otsu, roberts, sobel
from skimage.measure import regionprops, label
from skimage.exposure import equalize_adapthist
from skimage import io, exposure
from skimage.morphology import binary_dilation, remove_small_objects

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels
from deeplab_lfov.image_reader import ImageInput

from deeplab_lfov.tf_utils import chunks

#copy defaults over from train.py
from train import DATA_DIRECTORY,DATA_LIST_PATH,WEIGHTS_PATH,INPUT_SIZE
from train import load

from heatmap import Heatmap

#PARAMS
#####
SAVE_DIR = './heatmaps'
#RESTORE_FROM='./snapshots/model.ckpt-19999'
#RESTORE_FROM='./snapshots/model.ckpt-5000'
#RESTORE_FROM='./snapshots/model.ckpt-10000'
HR_RESTORE_FROM='./past_models/res0_model/snapshots/model.ckpt-9500'

HighResModel = DeepLabLFOVModel
#from deeplab_lfov.trivial_model import VGG
#from deeplab_lfov.trivial_model2 import VGG
#Model = VGG


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def hr_sess_saver(sess=None,var_list=None):
    if var_list is None:
        raise ValueError('pass hr_net.variables')
    if sess is None:
        # Set up tf session and initialize variables. 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Load weights.
    saver = tf.train.Saver(var_list=var_list)

    #restore_from needs to exist
    load(saver, sess, HR_RESTORE_FROM)

    return sess,saver

def pad_until_compatible(input_window, window_size, stride_size):
    '''
    input_window is array of boundaries (lwr, upr+1)
    it is padded until it is window_size

    Valid window_sizes: {win, win+stride, win+2*stride, win+3*stride}

    '''
    output_window=[]
    for bdry,win,stride in zip(input_window,window_size,stride_size):
        l,u=bdry
        assert(l<u)
        padd_to_add = stride - (u-l-win)%stride
        #if u-l+padd_to_add
        while u-l+padd_to_add<win:
            padd_to_add+=150
        l_pad_rad=np.floor(padd_to_add/2.).astype('int')
        u_pad_rad=np.ceil(padd_to_add/2.).astype('int')
        #out_bdry=np.array([l-l_pad_rad, u+u_pad_rad+1])
        out_bdry=np.array([l-l_pad_rad, u+u_pad_rad])
        out_bdry-= min(out_bdry.min(),0) #make positive
        output_window.append(out_bdry)
    return np.array(output_window)


##take in low res hmap and find largest CC to focus on

def get_hr_network( input_size ):
    with tf.name_scope("create_inputs"):
        #Doesn't actually read anything
        #just creates placeholder and preprocesses
        reader=ImageInput(
            ph_size=input_size,
            input_size=input_size)
    # Create network.
    net = HighResModel(reader.image_batch)#DeepLabLFOV
    return reader,net

def is_power2(num):
    'states if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)

def heatmap2boundary(
    H, #heatmap
    input_size,#size inputs to lfov model
    stride_size,#stride of lfov input
    hs,#downsampling of heatmap
    ):
    #assert is_power2(hs),'downsampling must be power of 2'


    #H is ~approx 50x50 in our dataset
    H #heatmap np array

    ##Unclear if otsu is conservative enough!!
    positives = np.array( H > threshold_otsu(H),dtype=np.uint8)
    #compare with (1-H)<= threshold_otsu(1-H)#not equivalent

    ###Pressing question: is it safe to rely on low res model to find the biggest CC??
    #for now just return largest CC without worry
        #else dialate, find cc, return multiple windows, let high res decide
            #Dialate before focusing on a region


    connective_regions, num = label(positives,background=0,return_num=True,connectivity=2)
    print 'num connected regions:',num
    assert connective_regions.max()<=255#np.int64
    connective_regions=connective_regions.astype(np.uint8)


    regions = regionprops(connective_regions,H)
    s_regions= sorted(regions,key= lambda r: -r.area )#sort big to small

    ##TODO: Is this method missing major parts!
    ##hack for now just pick biggest region and roll with that
    reg0=s_regions[0]
    largest_connective_region=np.array(connective_regions==reg0.label)


    #get bounding box::
        #input a CC: #output a box with rows,cols divisible by stride=(1/2)*input_size
    cc_rows, cc_cols=np.where(connective_regions == reg0.label)

    cc_bdry= hs * np.array([ [cc_rows.min(), cc_rows.max()],
                        [cc_cols.min(), cc_cols.max()] ]).astype('int')

    print 'ccbdry',cc_bdry
    win_bdry=pad_until_compatible(cc_bdry,input_size,stride_size)
    print 'winbdry',win_bdry

    return win_bdry

def sliding_window(image,window_size,stride_size):
    print image.shape, window_size, stride_size
    assert np.sum( (image.shape[:2]-window_size)%stride_size ) ==0,'evenly tile'
    I=image.shape
    W=window_size
    S=stride_size
    windows=[]
    tl_coord=[]
    for r in range(0, I[0]-W[0]+S[0], S[0]):
        for c in range(0, I[1]-W[1]+S[1], S[1]):
            tl_coord.append([r,c])
            windows.append( image[r:r+W[0],
                        c:c+W[1]])
    return tl_coord, windows



#High Res Heat Map
def hrhm(
    H,#heatmap

    #NOTE: possibly can get away with using getUChar since level=0
    whole_image,#the fully loaded high res image

    #optionally pass in TF stuff if it exists already in mem
    #edit:no longer optional
    net=None, #the high res net
    reader=None,
    sess=None,
    pat=None,
    node=None,
    sdir='.',


    input_size=[300,300],#size inputs to lfov model
    stride_size=[150,150], #stride of lfov input
    max_batch_size=16,
    hs=2400,#8*300 #downsampling of heatmap rel to highres image

    #restore_from='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/past_models/res0_model/snapshots/model.ckpt-9500',
    restore_from=HR_RESTORE_FROM,
            ):

    print 'running highres hmap!'


    input_size=np.array(input_size)
    stride_size=np.array(stride_size)

    #Get corresp image patch
    r_bdry, c_bdry = heatmap2boundary( H, input_size, stride_size, hs)
    patch_image=whole_image[r_bdry[0]:r_bdry[1],c_bdry[0]:c_bdry[1]]
    print 'lesion patch shape:',patch_image.shape

    patch_offset=[r_bdry[0],c_bdry[0]]

    #Save and Debug Image
    postr='_'+str(patch_offset[0])+'_'+str(patch_offset[1])+'_'
    fname='/highres_'+str(pat)+'_'+str(node)+postr
    ds=8#ds for saving downsampled images
    try:
        io.imsave(sdir+'/images'+fname+'.tif', patch_image)#corresp image
    except MemoryError:
        print 'MemoryError occured in saving High Res image..see downsampled img'
    io.imsave(sdir+'/images'+fname+'_downsampled.tif', patch_image[::ds,::ds])#corresp image

    top_left_coordinates, high_res_windows= sliding_window(patch_image,input_size,stride_size)
    batches_of_patches=chunks( [top_left_coordinates,high_res_windows],max_batch_size, axis=1 )


    #Create and Load tensorflow network.
    if net is None:
        print 'warning resetting default graph'
        tf.reset_default_graph()
        reader,net= get_network(input_size)
    image_batch=reader.image_batch

    # Which variables to load.
    #not the low res variables
    highres_var = net.variables

    # Predictions.
    net_pred = net.preds
    net_prob = net.probs

    if sess is None:
        # Set up tf session and initialize variables. 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.allow_soft_placement=True
        #config.log_device_placement=True
        sess = tf.Session(config=config)

    sess.run(tf.variables_initializer(highres_var))
    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())

    ## Load weights.
    #saver = tf.train.Saver(var_list=highres_var)
    #if restore_from is not None:
    #    print('restoring from ',restore_from)
    #    load(saver, sess, restore_from)

    #hr_net has output_size = input_size
    hr_hmap=Heatmap(patch_image.shape[:2])
    hr_hmap.half_window_stride=True

    # Iterate over images.
    step=0
    for bat_tl_coord,bat_patch in batches_of_patches:
        step+=1
        print 'step ',step

        if step==1:
            print 'running tf loop:hmhr!'
            print 'is batch_patch shape right??', bat_patch[0].shape

        probs = sess.run(net_prob, feed_dict={reader.ph_image: bat_patch})

        for tl_coord, prob_patch in zip(bat_tl_coord, probs):
            #coord already scaled by output_size
            hr_hmap.stitch(prob_patch[...,1], tl_coord )

    if not os.path.exists('./heatmaps'):
        os.mkdir('./heatmaps')
    if not os.path.exists('./images'):
        os.mkdir('./images')

    hr_hmap.save_img(sdir+'/heatmaps'+fname+'.tif')
    hr_hmap.save_img(sdir+'/heatmaps'+fname+'downsampled'+'.tif',ds=8)

    return hr_hmap


