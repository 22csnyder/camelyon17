"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from deeplab_lfov import  ImageReader, decode_labels
from deeplab_lfov.class_model import VGG

BATCH_SIZE = 16
DATA_DIRECTORY = '/mnt/nvme0n1p1/Data/CAMELYON_2017'
DATA_LIST_PATH = './dataset/train.txt'#pass this
#DATA_DIRECTORY = '/home/chris/Data' #'/home/VOCdevkit'
INPUT_SIZE = '300,300' #'321,321'
LEARNING_RATE = 1e-4
#TODO
#MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000
RANDOM_SCALE = True
RESTORE_FROM = ''#dont' restore by default
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/'
WEIGHTS_PATH   = None

#IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    #logdir = args.snapshot_dir
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(loader, sess, ckpt_path):
    '''Load trained weights.
    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

#def main():
if __name__ == '__main__':
    tf.reset_default_graph()
    """Create the model and start the training."""

    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            RANDOM_SCALE,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)

    # Create network.
    net = VGG( image_batch, label_batch )


    # Define the loss and optimisation parameters.
    loss = net.loss
    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimiser.minimize(loss, var_list=trainable)


    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=trainable, max_to_keep=40)


    #print('restore',args.restore_from)

    #Don't pass the whole file path
    #only pass the prefix:


    #default='snapshot/model.ckpt'
    #"python train.py --restore_from snapshots/model.ckpt-9500"
    if len(args.restore_from)>0:
        load(saver, sess, args.restore_from)
    else:
        print('Starting NEW model')

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    stophere


    print( 'Starting optimization..')


    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()

        #nothing fancy
        loss_value, _ = sess.run([loss, optim])

        if step % args.save_pred_every == 0:
            save(saver, sess, args.snapshot_dir, step)
        #    loss_value, images, labels,  _ = sess.run([loss, image_batch, label_batch, pred, optim])
        #else:
        #    loss_value, _ = sess.run([loss, optim])
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    save(saver, sess, args.snapshot_dir, step)
    coord.request_stop()
    coord.join(threads)

#if __name__ == '__main__':
#    main()
