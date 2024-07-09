'''
Input: a filepath
Work: makes two textfiles train.txt, val.txt in the same directory containing
the original file


#Take a list of image files, sorts them by patient


This is code that takes a list of filenames and paritions them into train and
val using 80/20 (or specified) split

It creates two new files: dataset/train.txt, dataset/val.txt

Please no spaces in filenames. I haven't coded for that
'''

import numpy as np
from utils import filename_parse
import os,itertools,sys

if __name__=='__main__':

    #textfile containing whole dataset
    input_fname = sys.argv[1]
    if not os.path.exists(input_fname):
        raise ValueError('could not find file',input_fname)

    dir_name=os.path.dirname(input_fname)
    train_fname=os.path.join(dir_name, 'train.txt')
    val_fname=os.path.join(dir_name, 'val.txt')

    if len(sys.argv)>=3:
        frac_train=sys.argv[2]
    else:
        frac_train=0.8

    #use randomseed
    if len(sys.argv)>=4:
        seed=sys.argv[3]
    else:
        seed=22
    np.random.seed(seed)

    input_f=open(input_fname)

    fn_dict=dict()
    #put different lines into different patient buckets
    with open(input_fname,'r') as f:
        for line in f:
            file1=line.split(' ')[0]
            pat,_=filename_parse(file1)
            if not pat in fn_dict.keys():
                fn_dict[pat]=[]
            fn_dict[pat].append(line)

    patients=fn_dict.keys()
    n_pats= len(patients)
    n_train_pats=int(np.round( frac_train * n_pats))

    np.random.shuffle(patients)#in place

    train_pats=np.sort(patients[:n_train_pats])
    val_pats=  np.sort(patients[n_train_pats:])

    def get_lines(pat_list):
        #maybe we want val lines to be sorted?eh
        lines=[]
        for pat in pat_list:
            lines.extend( fn_dict[pat] )
        return np.random.permutation(lines)#mix up order

    train_lines=get_lines( train_pats )
    val_lines  =get_lines(  val_pats  )

    with open(train_fname,'w') as train_f:
        train_f.writelines( train_lines )
    with open(val_fname,'w') as val_f:
        val_f.writelines( val_lines )



#    with open(input_fname,'r') as f:
#        n_input=np.sum( 1 for line in f )
#
#    n_train=int(np.round( frac_train * n_input))
#
#    np.random.seed(seed)
#    train_indices=np.random.choice( n_input, n_train, replace=False )
#
#    #the directory containing this file
#    project_dir= os.path.abspath(os.path.dirname(__file__))
#
#    #what to call the new train and val files
#    train_fname=os.path.join(project_dir, train_path)
#    val_fname=os.path.join(project_dir, val_path)
#
#    train_data=[]
#    val_data=[]
#
#    train_f = open(train_fname,'w')
#    val_f = open(val_fname, 'w')
#
#    with open(input_fname,'r') as f:
#        for i,line in enumerate(f):
#            if i in train_indices:
#                train_f.write(line)
#                #train_data.append( line )
#            else:
#                val_f.write(line)
#
#    train_f.close(),val_f.close()
#




