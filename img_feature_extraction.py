#image feature extraction

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

THRESHOLDS = [0.45, 0.5, 0.55, 0.6]
RMV_SIZES = [50, 60, 70, 80]
IMG_DIR = '/Users/kenleejr92/Desktop/heatmaps'
PKL_DIR = '/Users/kenleejr92/Desktop/pickles'
HEATMAP_FILE = '/Users/kenleejr92/Desktop/heatmaps/patient_020_node_4_heatmap.tif'
IMG_FILE='/Users/kenleejr92/Desktop/heatmaps/patient_020_node_4_heatmap.tif'

def get_arguments():
    parser=argparse.ArgumentParser(description='takes a directory of feature dictionaries and adds to them image features')
    parser.add_argument('--imgdir',type=str,default=IMG_DIR,
                       help='directory where images are stored')
    parser.add_argument('--pkldir',type=str,default=PKL_DIR,
                       help='directory of pickled feature dictionaries')
    parser.add_argument('--imgfile',type=str,default=IMG_FILE,
                       help='specific image file')
    return parser.parse_args()

#returns the count of small objects
def cell_counter(img, clip_limit, thresh, rmv_size, rmv_connectivity):
    #increase contrast
    img = equalize_adapthist(img, kernel_size = (50,50), clip_limit = clip_limit)
    gray_img = rgb2gray(img)
    binary = gray_img > thresh
    binary = remove_small_objects(binary, min_size=rmv_size, connectivity=rmv_connectivity)
    all_labels, num = measure.label(binary,return_num=True,background=0)
    return num

def create_blob_feature_vector(blob):
    cell_counts = []
    for t in THRESHOLDS:
        for rmv_size in RMV_SIZES:
            num = cell_counter(blob, 0.5, t, rmv_size ,2)
            cell_counts.append(num)
    return cell_counts

def append_img_features(picklefile,imgfile):
    '''
    appends RGB image features to end of dictionary
    '''
    with open(picklefile,'r') as f:
        data_dict = pickle.load(f)
        for key,val in data_dict.iteritems():
            features = val.values()[0]
            bbox = features[-5:]
            row = bbox[0]
            column = bbox[1]
            r_size = bbox[2]
            c_size = bbox[i]
            mask = bbox[4]
            if r_size == 0:
                print('no lesion')
                key1 = key
                key2 = val.keys()[0]
                data_dict[key1][key2] = features[:-5] + [0]

            else:
                pass
            '''
            #mlr code to get patch from original image
            reader = mir.MultiResolutionImageReader()
            mr_image = reader.open(imgfile)
            level = 0
            ds = mr_image.getLevelDownsample(level)
            image_patch = mr_image.getUCharPatch(column, row, r_size, c_size, level)
            zeros=(mask==0)
            image_patch[zeroes] = mask[zeroes]
            cell_counts = create_blob_feature_vector(image_patch)
            key1 = key
            key2 = val.keys()[0]
            data_dict[key1][key2] = features[:-5] + cell_counts
            '''
    with open(picklefile,'w+') as f:
        pickle.dump(data_dict,f)

def parse_pickle_file(pklfile):
    elements = pklfile.split('/')
    patient = elements[1]
    node=elements[3]
    return patient,node

if __name__ == '__main__':
    args=get_arguments()
    kwargs= vars(args)
    pklfiles = [args.pkldir+'/'+f for f in listdir(args.pkldir) if (isfile(join(arg.pkldir, f)) and f.endswith('.pkl'))]
    for f in pklfiles:
        found_file = False
        patient, node = parse_pickle_file(f)
        for root, dirs, files in os.walk(args.imgdir):
            for file in files:
                if file == 'patient_'+patient+'_node_'+node+'.tif':
                    found_file = True
                    imgfile = os.path.join(root, file))
                    append_img_features(f,imgfile)
        if not found_file: print('could not find corresponding image file')
    
