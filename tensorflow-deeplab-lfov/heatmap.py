"""
    This is built off the evaluation script. I try to take an image an make a
    heatmap for it.

    Evaluation script for the DeepLab-LargeFOV network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on around 1500 validation images.
"""


import argparse
from datetime import datetime
import os
import sys
import time
from skimage import io
import multiresolutionimageinterface as mir

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels
from deeplab_lfov.image_reader import ImageInput

from deeplab_lfov.tf_utils import chunks

#copy defaults over from train.py
from train import DATA_DIRECTORY,DATA_LIST_PATH,WEIGHTS_PATH,INPUT_SIZE
from train import load

#PARAMS
#####
SAVE_DIR = './heatmaps'
#RESTORE_FROM='./snapshots/model.ckpt-19999'
#RESTORE_FROM='./snapshots/model.ckpt-5000'
RESTORE_FROM='./snapshots/model.ckpt-10000'
LR_RESTORE_FROM='./snapshots/model.ckpt-10000'

#Model = DeepLabLFOVModel
#from deeplab_lfov.trivial_model import VGG
from deeplab_lfov.trivial_model2 import VGG
Model = VGG

####if we need to go down to level 2, this is where we do it!!##ds=4,
#half_image instead of whole_image
#ds=8


max_batch_size=128#16
####

#for inspection
#eg_filename='/mnt/md0/CAMELYON_2017/centre_2/patient_040_node_2.tif'
#eg_maskname='/mnt/md0/CAMELYON_2017/lesion_masks/patient_040_node_2_mask.tif'
#md_filename='/mnt/md0/CAMELYON_2017/centre_2/patient_040_node_2.tif'
#md_maskname='/mnt/md0/CAMELYON_2017/lesion_masks/patient_040_node_2_mask.tif'
#nvme_filename='/mnt/nvme0n1p1/Data/CAMELYON_2017/test_load_speed/patient_040_node_2.tif'
#nvme_maskname='/mnt/nvme0n1p1/Data/CAMELYON_2017/test_load_speed/patient_040_node_2_mask.tif'
#
#val_filename='/mnt/md0/CAMELYON_2017/centre_2/patient_042_node_3.tif'
#val_mask='/mnt/md0/CAMELYON_2017/lesion_masks/patient_042_node_3_mask.tif'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")

    #Heatmap arguments

    parser.add_argument("--input_image",type=str,
            help='location of histology image on which to do inference')

    #Deeplab arguments
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY, help="Path to the directory containing the PASCAL VOC dataset.")

    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Where to save predicted masks.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH, help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")
    return parser.parse_args()

def get_lr_network( input_size ):
    print 'resetting tf graph'
    tf.reset_default_graph()
    with tf.name_scope("create_inputs"):
        #Doesn't actually read anything
        #just creates placeholder and preprocesses
        reader=ImageInput(
            ph_size=input_size,
            input_size=input_size)
    # Create network.
    #net = DeepLabLFOVModel(weights_path)
    net = VGG(reader.image_batch)
    return reader,net


def lr_sess_saver(sess=None):
    #has to be done before get_hr_network
    if sess is None:
        # Set up tf session and initialize variables. 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # Load weights.
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    #restore_from needs to exist
    load(saver, sess, LR_RESTORE_FROM)
    return sess,saver


class Timer(object):
    def __init__(self):
        self.total_section_time=0.
        self.iter=0
    def on(self):
        self.t0=time.time()
    def off(self):
        self.total_section_time+=time.time()-self.t0
    def __str__(self):
        n_min=self.total_section_time/60.
        return '%.2fmin'%n_min

class Heatmap(object):
    '''
    Heatmap doesn't know anything about downsampling or global coordinates.
    It just assembles an array from patches
    '''
    half_window_stride=False

    def __init__(self,_shape):
        self._shape=_shape
        self.img= np.zeros(self.shape,dtype=np.float16)#f16 bc could get large

        self.min_so_far=np.array([np.inf,np.inf])#r,c
        self.max_so_far=np.array([-1,-1])
    @property
    def shape(self):
        return self._shape

    @property
    def prob_img(self):#divide by n times patch was sampled. should be prob_img\in[0,1]
        #TODO: fix hardcode

        #most of image hit 4 times
        prob_img=self.img/4.#3(s)

        #but border only twice
        #and corners only once
        prob_img[:150,:]*=2
        prob_img[:,:150]*=2
        prob_img[-150:,:]*=2
        prob_img[:,-150:]*=2

        return prob_img

    ##Consider Pillow. Image.paste()
    def stitch(self, patch, location):
        r,c = location
        r,c=int(r),int(c)

        #assume either scalar or 2D
        if len(patch.shape)==0:
            self.img[r,c] += patch
        elif len(patch.shape)==2:
            self.img[r:r+patch.shape[0],c:c+patch.shape[1]] += patch
        else:
            raise ValueError('patch needs len(shape)=2 or 4, but has',len(patch.shape))
        self.min_so_far=np.minimum(self.min_so_far, [r,c])
        self.max_so_far=np.maximum(self.max_so_far, [r,c])
    def save_img(self,fname,ds=1):
        print 'saving heatmap:',fname

        #whether to divide through
        if self.half_window_stride is True:
            img=self.prob_img
        else:
            img=self.img
        io.imsave(fname, img[::ds,::ds].astype(np.float32))


def get_whole_image(image_file,lvl=0):
    #assert lvl in [0,1]#lvl=1 -> half_image
    t0=time.time()
    reader = mir.MultiResolutionImageReader()
    mr_image= reader.open(image_file)
    ds=mr_image.getLevelDownsample(lvl)
    out_shape=np.round(np.array(mr_image.getDimensions())/ds)
    whole_image=mr_image.getUCharPatch(0,0,int(out_shape[0]),int(out_shape[1]),lvl)
    print 'lvl',lvl,' image read complete (',(time.time()-t0)/60.,'min)'
    return whole_image


def slice_shape(whole_shape,tile_size,ds):
    ds=int(ds)
    tile_size=np.array(tile_size).astype(np.int)
    range_tiles= (np.array(whole_shape[:2])//(ds*tile_size)).astype(np.int)
    print 'range_tiles',range_tiles
    slices,coords=[],[]
    for i in range(range_tiles[0]):
        for j in range(range_tiles[1]):
            row=i*ds*tile_size[0]
            col=j*ds*tile_size[1]

            slices.append([i,j,slice(i*ds*tile_size[0],(i+1)*ds*tile_size[0],ds),
                        slice(j*ds*tile_size[1],(j+1)*ds*tile_size[1],ds)])
            coords.append([row,col,ds])
    return slices, coords



#def main(whole_image=None,net=None,reader=None,sess=None,pat=None,node=None,**kwargs):
def main(whole_image=None,net=None,reader=None,sess=None,pat=None,node=None,sdir='.',ds=1):
    if not pat:
        raise ValueError('pass pat, node')
    fname='/lowres_'+str(pat)+'_'+str(node)+'_ken'

    #tf.reset_default_graph()

    """Create the model and start the evaluation process."""

    #save_dir=kwargs['save_dir']
    input_size='300,300'
    #input_size=kwargs['input_size'] #also tile_size
    h, w = map(int, input_size.split(','))
    input_size = np.array([h, w])


    if whole_image is None:
        raise ValueError('pass whole_image')
        #input_image=kwargs['input_image']
        #whole_image=get_whole_image(input_image)


    tile_slices,tile_coord=slice_shape(whole_image.shape, input_size,ds)
    batches_of_patches=chunks( [tile_slices,tile_coord],max_batch_size, axis=1 )


    #Create and Load tensorflow network.
    if net is None:
        reader,net= get_lr_network(input_size)
        image_batch=reader.image_batch

    # Predictions.
    net_pred = net.preds
    net_prob = net.probs

    # mIoU
    #mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, label_batch, num_classes=2)

    ## Set up tf session and initialize variables. 
    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Load weights.
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        #restore_from needs to exist
        load( saver, sess,LR_RESTORE_FROM )


    #needed?
    batch_prob_shape=net.probs.get_shape().as_list()
    if len(batch_prob_shape)==2:
        output_size=np.array([1.,1.])
    elif len(batch_prob_shape)==4:
        output_size=np.array(batch_prob_shape[1:3]).astype('int')
    else:
        raise ValueError('prob should be either 2 or 4 dim')


    #ideally this is an integer(array)
    hs= ds*input_size.astype('float')/output_size
    #print ds, '\n',input_size,'\n',output_size

    heatmap_size= (np.array(whole_image.shape[:2]) // hs).astype('int')
    hmap=Heatmap(heatmap_size)
    hmap.half_window_stride=False


    #Timers
    tgen, ttf,thmap=Timer(),Timer(),Timer()
    tnp=Timer()


    # Iterate over images.
    step=0
    for batch_slice,batch_coord in batches_of_patches:
        step+=1
        if step==1:
            print 'running tf loop:lowrew heat map!'

        print 'step ',step
        tgen.on()
        brows, bcols, br_slices, bc_slices = zip(*batch_slice)
        tgen.off()

        tnp.on()
        batch_img=np.stack([whole_image[rsli,csli] for rsli,csli in
                            zip(br_slices, bc_slices)])
        tnp.off()

        ttf.on()
        probs = sess.run(net_prob, feed_dict={reader.ph_image: batch_img})
        ttf.off()

        thmap.on()
        for r,c,patch in zip(brows,bcols,probs):
            hmap.stitch(patch[...,1],[r*output_size[0], c*output_size[1]] )
        thmap.off()

        if step % 100 == 0:
            print('step {:d} \t'.format(step))

    print tgen, tnp, ttf, thmap
    #io.imsave(save_dir+'/'+'hmap_trivial_40_2.tif',hmap.img)

    if not os.path.exists('./heatmaps'):
        os.mkdir('./heatmaps')
        #ckpt=kwargs['restore_from'].split('-')[-1]
        #ckiter=ckpt.split('.')[0]
    #hmap.save_img('./heatmaps/hmap_trivial_40_2'+ckiter+'.tif')
    #hmap.save_img('./heatmaps/val_42_3_hmap_trivial2_iter'+ckiter+'.tif')
    #hmap.save_img('./heatmaps/val_42_3_hmap_trivial2_iter'+ckiter+'.tif')
    hmap.save_img(sdir+'/heatmaps'+fname+'.tif')

    return hmap

#maybe helpful later
    #mIoU_value = sess.run([mIoU])
    #_ = update_op.eval(session=sess)


if __name__ == '__main__':

    args=get_arguments()
    kwargs= vars(args)
    main( **kwargs )

