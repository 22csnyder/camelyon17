import numpy as np
import pandas as pd
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

TRAIN_DICT='/Users/kenleejr92/Desktop/pickles/train.pkl'
TEST_DICT='/Users/kenleejr92/Desktop/pickles/test.pkl'
NEG_LIST='/Users/kenleejr92/Desktop/pickles/test_neg.pkl'
OUTPUT_DIR='/Users/kenleejr92/Desktop/'

def get_arguments():
    parser=argparse.ArgumentParser(description='trains a classifier on heatmap features and predicts for given test data')
    parser.add_argument('--train_dict',type=str,default=TRAIN_DICT,
                       help='data dictionary of features and targets for training heatmaps')
    parser.add_argument('--test_dict',type=str,default=TEST_DICT,
                       help='data_dictionary of features for testing heatmaps')
    parser.add_argument('--neg_list',type=str,default=NEG_LIST,
                       help='list of patient_node that are labeled negative')
    parser.add_argument('--outputdir',type=str,default=OUTPUT_DIR,
                        help='directory where .csv result file is written')
    return parser.parse_args()


def create_result_csv(y_pred,index,patient_dict,outputdir,encoder):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    with open(outputdir+'results.csv','w+') as f:
        class_dict = {'negative':0,'itc':0,'macro':0,'micro':0}
        for i, y_pred in enumerate(y_pred):
            patient, node = index[i].split('_')
            node_int = int(node)
            patient_dict[patient][node_int]=y_pred
        for key in patient_dict:
            unique, counts = np.unique(patient_dict[key], return_counts=True)
            unique = unique.astype(dtype=int)
            unique = encoder.inverse_transform(unique)
            for i,u in enumerate(unique): class_dict[u]=counts[i]
            metastasis_counts = class_dict 
            print(metastasis_counts)
            patient_class = ''
            metastasis_counts['meta'] = metastasis_counts['macro'] + metastasis_counts['micro']
            if(metastasis_counts['negative'] == 5): patient_class = 'pN0'
            elif(metastasis_counts['micro'] > 0 and metastasis_counts['macro'] == 0): patient_class = 'pN1mi'
            elif(metastasis_counts['meta'] > 1 and metastasis_counts['meta'] < 3 and metastasis_counts['macro']>1): patient_class = 'pN1'
            elif(metastasis_counts['meta'] > 3 and metastasis_counts['macro']>1): patient_class = 'pN2'
            else: patient_class = 'pN0(i+)'
            f.write('patient_'+patient+'.zip'+','+patient_class+'\n')
            node_classes = patient_dict[key]
            node_classes = node_classes.astype(dtype=int)
            node_classes = encoder.inverse_transform(node_classes)
            for node_class in node_classes:
                f.write('patient_'+patient+'_node_'+node+'.tif'+','+node_class+'\n')

def create_data_matrix(data_dict,negative):
    patient_dict = {}
    feature_dict = {}
    target_dict = {}
    for key,val in data_dict.iteritems():
        patient, node = key.split('_')
        feature_dict[key] = val.values()[0]
        target_dict[key] = val.keys()[0]
        if patient not in patient_dict: patient_dict[patient]=np.ones((5,))*negative
    feature_frame = pd.DataFrame.from_dict(feature_dict,orient='index')
    target_frame = pd.DataFrame.from_dict(target_dict,orient='index')
    target_frame.columns=['target']
    return feature_frame, target_frame, patient_dict

if __name__ == '__main__':
    args=get_arguments()
    kwargs= vars(args)

    #encode lymph node classes
    enc = LabelEncoder()
    lesion_labels = ['negative','itc','macro','micro']
    enc.fit(lesion_labels)
    codes = enc.transform(lesion_labels)
    negative = codes[0]

    with open(args.train_dict,'r') as f1:
        train_dict = pickle.load(f1)
    with open(args.test_dict,'r') as f2:
        test_dict = pickle.load(f2)
    with open(args.neg_list,'r') as f3:
        neg_list = pickle.load(f3)
         
    train_fframe,train_tframe,_ = create_data_matrix(train_dict,negative)
    test_fframe,_,patient_dict = create_data_matrix(test_dict,negative)
    data_frame = train_fframe.join(train_tframe,lsuffix='',rsuffix='')
    data_frame['target'] = enc.transform(data_frame['target'])

    train = data_frame.as_matrix()
    x_train = train[:,:-1]
    y_train = train[:,-1]
    
    print data_frame
    #Train classifier
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)

    x_test = test_fframe.as_matrix()
    test_index = test_fframe.index
    y_pred = rf.predict(x_test)

    for nl in neg_list:
        patient,node=nl.split('_')
        patient_dict[patient][int(node)] = negative

    create_result_csv(y_pred,test_index,patient_dict,args.outputdir,enc)


