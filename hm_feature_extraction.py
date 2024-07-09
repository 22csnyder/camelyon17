'''
Takes in heatmap of lymph node produced by NN and predicts metastasis class
Must classify each "blob"
Metastasis classes for lymph nodes:
    negative: no tumor cells
    itc: single tumor cell or cluster < 0.2 mm or less than 200 cells
    micro: 0.2 mm or more than 200 cells but < 2.0 mm
    macro: greater than 2.0 mm

Patient classes:
    pN0: No micro-metastases or macro-metastases or ITCs found.
    pN0(i+): Only ITCs found.
    pN1mi: Micro-metastases found, but no macro-metastases found.
    pN1: Metastases found in one to three lymph nodes, of which at least one is a macro-metastasis.
    pN2: Metastases found in four to nine lymph nodes, of which at least one is a macro-metastasis.
Takes in a directory of heat maps: 1-channel

blob feature ideas:
    cell count: can't do without original image
    % of tumor over whole tissue region
    % of tumor over surrouding convex region (blob)
    average prediction values over tumor
    variance of prediction values
    longest axis of tumor

    will have to have dictionary for each patient and node that holds the feature vectors
    then go through dictionary and pick largest one
'''

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

HEATMAP_DIR='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/heatmaps'
IMAGE_DIR='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/images'
GT_FILE='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/stage_labels.csv'
DUMP_DIR = '/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/pickles/'
HEATMAP_FILE='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/heatmaps/highres_42_3_62325_31125_.tif'
IMAGE_FILE='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/images/highres_42_3_62325_31125_.tif'


def get_arguments():
    parser=argparse.ArgumentParser(description='takes a directory of heatmaps and predicts lymph node classes')
    parser.add_argument('--heatmapdir',type=str,default=HEATMAP_DIR,
                       help='directory of heatmap files')
    parser.add_argument('--testdir',type=str,default=HEATMAP_DIR,
                       help='directory of test heatmap files')
    parser.add_argument('--heatmapfile',type=str,default=HEATMAP_FILE,
                       help='specific heatmap file')
    parser.add_argument('--gtfile',type=str,default=GT_FILE,
                        help='.csv containing ground truth')
    parser.add_argument('--dumpdir',type=str,default=DUMP_DIR,
                        help='directory to save data dictionary')
    parser.add_argument('--imgfile',type=str,default=IMAGE_FILE,
                        help='image file')
    return parser.parse_args()

#the count of small objects
def cell_counter(img, thresh, rmv_size):
    #increase contrast
    img = equalize_adapthist(img, kernel_size = (50,50), clip_limit = 0.5)
    gray_img = rgb2gray(img)
    binary = gray_img > thresh
    binary = remove_small_objects(binary, min_size=rmv_size, connectivity=2)
    all_labels, num = measure.label(binary,return_num=True,background=0)
    return num

def create_blob_feature_vector(blob):
    cell_counts = []
    for t in THRESHOLDS:
        for rmv_size in RMV_SIZES:
            num = cell_counter(blob, t, rmv_size)
            cell_counts.append(num)
    return cell_counts

def extract_features_heatmap(patient,
                            node,
                            heatmap,
                            image,
                            pn_dict,
                            no_lesions,
                            include_img_features=False,
                            gtfile=None,
                            train=True,
                            threshold=0.25):
    '''
    extracts features from a heatmap and adds to dictionary
    #######parameters#######
    patient: string
    node: string
    heatmap: numpy array, 1-channel
    image: numpy array, 3-channel
    pn_dict: dictionary
        key1 = 'patient_node'
        key2 = class label if train, 'test' if test
        value = feature vector
    no_lesions = list
        list of patient_nodes that are negative
    include_img_features:
        if True, appends a bounding box and mask to feature vector
        the bounding box is used by img_feature_extraction to
        append features from the RGB image
    gtfile: name of file holding ground truths
    train: boolean, whether training or testing heatmap
    threshold: probability threshold
    '''
    #print('Threshold WARNING: Chris says heat map will be in [0,1]. Also it will be small enough to do otsu. You can throw an exception if its bigger than (20000,20000)')

    lab=None
    if train:
        lines = [line.rstrip('\r\n') for line in open(gtfile)]
        for line in lines:
            filename,lbl = line.split(',')
            if filename == 'patient_'+patient+'_node_'+node+'.tif':
                lab = lbl
        if lab==None: print('no lesion label')

    #threshold by probability
    positives = np.array(heatmap > threshold, dtype=np.uint8)
    print('finding connective regions')
    #get connected regions
    connective_regions, num = label(positives,background=0,return_num=True,connectivity=2)
    print('number of connected regions: %d' % num)
    if num == 0:
        #predict negative
        print('no lesion')
        no_lesions.append(patient+'_'+node)
        return no_lesions
    else:
        #get features of all connected regions, and take only largest one
        regions = regionprops(connective_regions,heatmap)
        max_region = 0.0
        max_idx = 0
        max_label = 0
        feature_dict = {}
        feature_vector = []
        print('finding largest connective component')
        for idx in range(len(regions)):
            if regions[idx]['area'] > max_region and regions[idx]['label']!=0:
                max_region = regions[idx]['area']
                max_label = regions[idx]['label']
                max_idx = idx
        print('masking largest connective component')
        largest_connective_region = np.array(connective_regions==regions[max_idx]['label'])
        print('making feature vector')
        #append appropriate features to feature vector
        feature_vector = [regions[max_idx]['area'],
                            regions[max_idx]['eccentricity'],
                            regions[max_idx]['equivalent_diameter'],
                            regions[max_idx]['euler_number'],
                            regions[max_idx]['extent'],
                            regions[max_idx]['filled_area'],
                            regions[max_idx]['inertia_tensor_eigvals'][0],
                            regions[max_idx]['inertia_tensor_eigvals'][1],
                            regions[max_idx]['major_axis_length'],
                            regions[max_idx]['max_intensity'],
                            regions[max_idx]['mean_intensity'],
                            regions[max_idx]['min_intensity'],
                            regions[max_idx]['minor_axis_length'],
                            regions[max_idx]['perimeter']]

        if include_img_features:
            print('appending image features')
            min_row, min_column, max_row, max_column  = regions[max_idx]['bbox']
            r_size = max_row-min_row
            c_size = max_column-min_column
            bbox = connective_regions[min_row:min_row+r_size,min_column:min_column+c_size]
            lcr_mask = (bbox!=regions[max_idx]['label'])
            sub_image = image[min_row:min_row+r_size,min_column:min_column+c_size,:]
            zeros = np.ones((r_size,c_size,3))
            sub_image[lcr_mask,:] = zeros[lcr_mask]
            io.imshow(sub_image)
            io.show()
            ################not doing cell counts until later#############
            #cell_counts = count_cells(sub_image) 
            feature_vector = feature_vector + [50]

        #append target label if training
        print('creating dictionary')
        if train:
            feature_dict[lab]=feature_vector
        else:
            feature_dict['test']=feature_vector

        pn=patient+'_'+node
        pn_dict[pn] = feature_dict
        return pn_dict

def save_pickle(outputdir,pkl,train_or_test):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    with open(outputdir+train_or_test+'.pkl','w+') as f:
        pickle.dump(pkl,f)

if __name__ == '__main__':
    args=get_arguments()
    kwargs= vars(args)
    include_img_features = False
    train_dict={}
    test_dict={}
    no_lesions_test = []
    no_lesions_train = []
    print('reading heatmap')
    heatmap = io.imread(args.heatmapfile)
    print('reading img')
    imgfile = io.imread(args.imgfile)
    extract_features_heatmap('042','3',
                            heatmap,imgfile,
                            train_dict,no_lesions_train,
                            include_img_features,
                            args.gtfile,
                            train=True)
#    extract_features_heatmap('042','3',
#                            heatmap,imgfile,
#                            test_dict,no_lesions_test,
#                            include_img_features,
#                            args.gtfile,
#                            train=False)
    print('done extracting features')
    save_pickle(args.dumpdir,train_dict,'train')
#    save_pickle(args.dumpdir,test_dict,'test')
    save_pickle(args.dumpdir,no_lesions_train,'train_neg')
 #   save_pickle(args.dumpdir,no_lesions_test,'test_neg')
    print(train_dict)
    print(no_lesions_train)

