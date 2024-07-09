
import glob
import os
from heatmap import *
from low2highres import *
input_size=[300,300]

'''
random stuff to facilitate going through test data
making heatmaps
and maybe generating heatmap features

'''

def is_int(s):
    try:
        int(s)
        return True
    except:
        return False
def filename_parse(fname):
    '''
    returns patient_number, node_number that identify the data
    It finds the first two strings that can be converted to numbers that are
    flanked by underscores: _#_
    '''
    #remove trailing / from dir
    if fname.endswith('/'):
        fname=fname[:-1]

    #remv ext if exists
    fname=os.path.splitext(fname)[0]
    base=os.path.basename(fname)
    parse = base.split('_')#first two numbers
    patient, node = filter(is_int, parse)[:2]
    return int(patient),int(node)


#test_dir='/mnt/md0/test_data/centres'


#test_files=sorted( glob.glob(test_dir+'/*.tif'),key=lambda f: filename_parse(f))

'''These are the two main files that have all the functions to call in ipython
to make the inference work
'''
TRAIN_IMG = '/mnt/md0/CAMELYON_2017/centre_1/patient_039_node_1.tif'
IN_DIR = '/media/chris/Untitled/CAMELYON_2017/centre*'
OUT_DIR = './train_outputs'
def get_arguments():
    parser=argparse.ArgumentParser(description='training_file')
    parser.add_argument('--outputdir',type=str,
                        default=OUT_DIR, help='directory of training files')
    parser.add_argument('--inputdir',type=str,
                        default=IN_DIR,help='')
    return parser.parse_args()

def run_low_res_model(filename,outputdir,lr_reader,lr_net,sess):
    whole_image=get_whole_image(filename,lvl=3)
    pat,node=filename_parse(filename)
    lr_hmap=main( whole_image, lr_net, lr_reader, sess, pat,node,outputdir)


if __name__ == '__main__':
    args=get_arguments()
    lr_reader, lr_net = get_lr_network( [300,300] )
    sess,lr_saver=lr_sess_saver()
    for filename in glob.iglob(args.inputdir+'/*.tif'):
        print('ken',filename)
        try:
            run_low_res_model(filename,args.outputdir,lr_reader,lr_net,sess)
        except:
            print('error opening ' + filename.split('/')[-1])
            continue
    sess.close()
