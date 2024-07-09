import os

import numpy as np
import tensorflow as tf

import argparse
from deeplab_lfov import ImageReader

from train import get_arguments


from train import INPUT_SIZE, DATA_DIRECTORY,DATA_LIST_PATH
#INPUT_SIZE = '300,300' #'321,321'
#DATA_DIRECTORY = '/home/chris/Data' #'/home/VOCdevkit'
#DATA_LIST_PATH = './dataset/first_train.txt' #'./dataset/train.txt'


parser = argparse.ArgumentParser(description="DeepLabLFOV Network")

parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the PASCAL VOC dataset.")
parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                    help="Path to the file listing the images in the dataset.")
parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                    help="Comma-separated string with height and width of images.")


RANDOM_SCALE = False

#def main():

if __name__ == '__main__':
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
            coord,
            num_epochs=1)#only go for 1 epoch then stop

    mean_image=tf.contrib.metrics.streaming_mean( reader.image )


    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())#for streaming mean
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    ##Iterate through data once##
    i=0
    with coord.stop_on_exception():
        while not coord.should_stop():
            i+=1
            X_mean, X= sess.run( [mean_image, reader.image] )

    coord.request_stop()
    coord.join(threads)



