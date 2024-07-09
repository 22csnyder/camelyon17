"""Evaluation script for the CAMELYON Dataset

probs_stitch is a function that takes a text_file specifying sample images from a specific
patient and node. Creates a probability heatmap by calculating pixel probabilities and placing
them on a template the same size as the original image
"""

from __future__ import print_function

import pprint
import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image
from skimage import io

import tensorflow as tf
import numpy as np

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels
import multiresolutionimageinterface as mir

pp=pprint.PrettyPrinter()


#The DATA_DIRECTORY should contain sliding window images from a single patient and node.
#So that when we perform prediction, we can work with one template to which we add probabilities
#probs_stich should be called once per patient per node. 

#data directory is what contains the folder CAMELYON_2017 which contains training_samples,
#which contains patient directories
#training.txt determines what files will be evaluated, now we have two text files, one that has all samples together, and
#one for each patient and node


def get_arguments():
    parser = argparse.ArgumentParser(description="prob_stitch")
    parser.add_argument("--inference_dir", type=str,
                        help="directory full of slide-window tif images")
    parser.add_argument("--restore_from", type=str,
                        help="Path to the ckpt to restore from")
    parser.add_argument("--save_dir", type=str,
                        help="where output heatmap is saved")
    return parser.parse_args()


#def get_arguments():
#def probs_stitch(data_directory = '/home/kenleejr92/Desktop',\
#                 data_list_path = '/home/kenleejr92/Desktop/text_files/patient_017_node_2.txt',\
#                 restore_from = '/home/kenleejr92/Desktop/snapshots/model.ckpt-8500',\
#                 save_dir = '/home/kenleejr92/Desktop/predictions/',\
#                 output_dir = '/home/kenleejr92/Desktop',\
#                 weights_path  = None,\
#                 num_steps = 1):


def decode_imgfilename(filename):
    fname = filename.split('/')[-1]
    elements = fname.split('_')
    patient = elements[1]
    node = elements[2]
    y = int(elements[3])
    x = int(elements[4].split('.')[0])
    return patient, node, y, x

def decode_txtfilename(filename):
    print(filename)
    fname = filename.split('/')[-1]
    print(fname)
    elements = fname.split('_')
    patient = elements[1]
    node = elements[3].split('.')[0]
    return patient, node

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

class Heatmap(object):
    def __init__(self,r,c):
        self.r=r
        self.c=c
        self.img= np.zeros((r+300,c+300),dtype=np.float16)

        self.min_so_far=np.array([np.inf,np.inf])#r,c
        self.max_so_far=np.array([-1,-1])

    ##Consider Pillow. Image.paste()
    def stitch(self, patch, location):
        r,c = location
        self.img[r:r+patch.shape[0],c:c+patch.shape[1]] += patch

        self.min_so_far=np.minimum(self.min_so_far, [r,c])
        self.max_so_far=np.maximum(self.max_so_far,
                                   [r+patch.shape[0],c+patch.shape[1]])


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


#def probs_stitch(inference_dir='/mnt/nvme0n1p1/Data/CAMELYON_2017/patient_020_node_4',
#                 data_dir='/media/chris/Untitled/CAMELYON_2017',
#                 restore_from = '/home/kenleejr92/Desktop/snapshots/model.ckpt-8500',
#                 save_dir = '/home/kenleejr92/Desktop/predictions/',
#                 weights_path  = None,
#                 num_steps = 1):
if __name__=='__main__':
    tf.reset_default_graph()
    inference_dir='/mnt/nvme0n1p1/Data/CAMELYON_2017/patient_020_node_4'
    restore_from='snapshots/model.ckpt-9500'
    save_dir='heatmaps'
    input_size='300,300'
    weights_path=None


    h, w = map(int, input_size.split(','))
    input_size = (h, w)

    """
    This takes in a directory of jpg files, where each of the files is some
    sliding window of a very large tif image.
    Find patient and node, determine size, create numpy template to add to
    """

    #patient, node = decode_txtfilename(inference_dir)
    #print(patient,node)
    #pfile=os.path.join(data_dir, 'patient_%s_node_%s.tif' % (patient,node)
    #try:
    #    filepath = find_file(pfile, data_directory)
    #    print(filepath)
    #    reader = mir.MultiResolutionImageReader()
    #    mr_image = reader.open(filepath)
    #    c,r = mr_image.getDimensions()
    #    print(c,r)
    #except:
    #    print('file not found %s' % (pfile))
    #    sys.exit(1)

    #just hardcode..
    r=100000
    c=220000
    ################Lots of memory used#################
    print('start init template..')
    heatmap=Heatmap(r,c)
    print('..finish init template')
    ###################################################

    """Create the model and start the evaluation process."""
    #args = get_arguments()

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    #batch_size=100
    batch_size=16

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            inference_dir=inference_dir,
            #input_size=None,#don't rescale or anything
            input_size=input_size,#don't rescale or anything
            random_scale=False,
            shuffle=False,
            coord=coord,
            num_epochs=1)
        #image, label = reader.image, reader.label
        #image_batch= tf.expand_dims(image, dim=0)# Add the batch dimension.

        #image, image_file = reader.inf_image, reader.inf_image_file

        image_batch, image_file_batch = reader.dequeue_inference(batch_size)


    # Create network.
    #with tf.device('/gpu:0'):
    #    net = DeepLabLFOVModel(weights_path)
    with tf.device('/gpu:5'):
        net = DeepLabLFOVModel(weights_path)
    #net = DeepLabLFOVModel(weights_path)

    # Which variables to load.
    trainable = tf.trainable_variables()

    # Predictions.
    net_pred = net.preds(image_batch)
    net_prob = net.probs


    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    config.log_device_placement=True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Load weights.
    saver = tf.train.Saver(var_list=trainable)
    if restore_from is not None:
        print('restoring from ',restore_from)
        load(saver, sess, restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    i=0
    tstart=time.time()
    # Iterate over images.

    t_tf_total=0
    t_np_total=0
    with coord.stop_on_exception():
        while not coord.should_stop():
            #i+=1

            tsess=time.time()
            prob, pred, fname_batch=sess.run([net_prob,net_pred,image_file_batch])
            #prob, pred, fname = sess.run([net_prob,net_pred,image_file])
            t_tf_total+=(time.time()-tsess)

            fname=fname_batch[0]
            patient, node, y, x = decode_imgfilename(fname)
            #print('pt',patient,'node',node,'pos',y,x)
            #print('predshape',pred.shape)#(1,300,300,1)

            if i==0:
                print(len(prob),'samples per loop')
                print('probshape',prob.shape)#(1,300,300,2)
            i+=len(prob)

            patch=prob[0,:,:,1]

            tnp=time.time()
            heatmap.stitch(patch,(y,x))
            t_np_total+=(time.time()-tnp)

            #print('prediction scores')
            #print(net_pred.shape)
            #print('probabilities')
            #print(net_prob[0,0,:,:,:].shape)

            if i>=100:
                coord.request_stop()#debug
                print('iter ',i)
                tot=time.time()-tstart
                print('totaltime:',tot)
                print('t per sample:',tot/i)
            #################Lots of memory#################3
            #template = img_stitch(template, probs[0,0,:,:,:], (x,y))
            #################################################

    print('tf_time:',t_tf_total)
    print('np_time:',t_np_total)
    coord.join(threads)


    #TODO: update to save here
    if save_dir is not None:
        #heatmap = template[:,:,1]
        #io.imsave(heatmap,output_dir+'/patient_%s_node_%s_heatmap.jpg' % (patient,node))
        #predicted = np.argmax(template,axis=2)
        #io.imsave(predicted,output_dir+'/patient_%s_node_%s_pred.jpg' % (patient,node))
        #save numpy array?
        pass

    #return heatmap


#if __name__ == '__main__':
#    tf.reset_default_graph()
#    args=get_arguments()
#    kwargs= vars(args)
#    hmap=probs_stitch(**kwargs)

