from skimage import io, exposure
from skimage.morphology import binary_dilation, remove_small_objects
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, roberts, sobel
from skimage.color import rgb2gray
from skimage.feature import canny, blob_dog
from skimage.measure import regionprops, label
from skimage import measure
from skimage.exposure import equalize_adapthist
from os import listdir
from os.path import isfile, join, basename
import time
import os, sys
import argparse
from skimage.data import binary_blobs
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import cPickle as pickle

PKL_DIR = '/Users/kenleejr92/Desktop/pickles'
DUMP_DIR='/Users/kenleejr92/Desktop/random_forest/'

def get_arguments():
    parser=argparse.ArgumentParser(description='takes a directory of heatmaps and predicts lymph node classes')
    parser.add_argument('--pkldir',type=str,default=PKL_DIR,
                       help='directory of individual pickled features')
    parser.add_argument('--dumpdir',type=str,default=DUMP_DIR,
                       help='where to store test and train dictionaries')
    return parser.parse_args()

if __name__ == '__main__':
    args=get_arguments()
    kwargs= vars(args)

    train_dict={}
    test_dict={}

    files = [args.pkldir+'/'+f for f in listdir(args.pkldir) if (isfile(join(args.pkldir, f)))]
    for f in files:
        if f.endswith('train.pkl'):
            with open(join(args.pkldir,f),'r') as f1:
                feature_dict = pickle.load(f1)
                train_dict[feature_dict.keys()[0]] = feature_dict.values()[0]
        if f.endswith('test.pkl'):
            with open(join(args.pkldir,f),'r') as f2:
                feature_dict = pickle.load(f2)
                test_dict[feature_dict.keys()[0]] = feature_dict.values()[0]


    with open(args.dumpdir+'train_dict.pkl','w+') as f:
        pickle.dump(train_dict,f)

    with open(args.dumpdir+'test_dict.pkl','w+') as f:
        pickle.dump(test_dict,f)
