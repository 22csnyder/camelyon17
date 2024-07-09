import argparse
from datetime import datetime
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')#needed?
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage import io


from deeplab_lfov import decode_labels

from train import DATA_DIRECTORY,DATA_LIST_PATH,RESTORE_FROM,WEIGHTS_PATH,INPUT_SIZE
data_dir=DATA_DIRECTORY
input_size=INPUT_SIZE
weights_path=WEIGHTS_PATH


from train import load
###
from heatmap import get_network
###
'''
This is a convenient to load some validation patches and evaluates how the
model performs
'''

if __name__=='__main__':
    tf.reset_default_graph()

    h, w = map(int, input_size.split(','))
    input_size = (h, w)

    #frequently changed parameters#
    restore_from='./snapshots/model.ckpt-500'
    #restore_from='./snapshots/model.ckpt-19500'#always prob=0.2412
    n_examples=10
    val_fname='/mnt/nvme0n1p1/Data/CAMELYON_2017/training_samples_res3/text_files/val.txt'
    #frequently changed parameters#


    ##Get Data
    val_lines=[]
    with open(val_fname,'r') as val_f:
        for line in val_f:
            val_lines.append(line)

    np.random.shuffle(val_lines)

    val_half_fn=[ line.strip('\n').split(' ') for line in val_lines[:n_examples] ]
    val_fn=[ [data_dir+'/'+fi, data_dir+'/'+fm] for fi,fm in val_half_fn]

    info=[]
    for image_file, mask_file in val_fn:
        data=[]
        image=io.imread(image_file)
        mask=io.imread(mask_file)

        data.append(image)
        data.append(mask)

        info.append(data)
        #info.append([image,mask,prob,logit,pred])


    image_list=zip(*info)[0]
    mask_list=zip(*info)[1]




    ##Get Model
    reader,net=get_network(weights_path, input_size)


    net_pred = net.preds(reader.image_batch)
    net_prob = net.probs

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    trainable = tf.trainable_variables()
    saver = tf.train.Saver(var_list=trainable)
    load(saver, sess, restore_from)


    fd={reader.image_batch: np.stack(image_list)}

    batch_prob= sess.run( net_prob, feed_dict=fd )

    for i,prob in enumerate(batch_prob):
        pimg=np.squeeze(prob[:,:,1])
        io.imsave('./heatmaps/prob_image'+str(i)+'.tif', pimg)
    prob_list=[np.squeeze(prob[:,:,1]) for prob in batch_prob]


    #titles=['data','mask','prob','logit','pred']
    titles=['data','mask','prob']
    info=[
        [im[:,:,::-1] for im in image_list],
        [decode_labels(m) for m in mask_list],
        prob_list
        ]

    n_cols=len(titles)
    fig, axes = plt.subplots(n_examples, n_cols, figsize = (16, 12))
    for i in xrange(n_examples):
        for j in range(n_cols):
            if i==0:
                axes.flat[i * n_cols + j].set_title(titles[j])
            axes.flat[i * n_cols + j].imshow(info[j][i])

        #axes.flat[i * n_cols].imshow((images[i])[:, :, ::-1].astype(np.uint8))
        #axes.flat[i * n_cols + 1].imshow(decode_labels(labels[i, :, :, 0]))
        #axes.flat[i * n_cols + 2].imshow(decode_labels(preds[i, :, :, 0]))
    plt.savefig('./heatmaps/debug_model_500.png')
    plt.close(fig)




