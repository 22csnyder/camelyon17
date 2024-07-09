import glob
import os

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
from heatmap import *
from low2highres import *
input_size=[300,300]

#TRAIN_IMG = '/mnt/md0/CAMELYON_2017/centre_2/patient_042_node_3.tif'#good
TRAIN_IMG = '/mnt/md0/CAMELYON_2017/centre_1/patient_036_node_3.tif'

def get_arguments():
    parser=argparse.ArgumentParser(description='training_file')
    parser.add_argument('--train_img',type=str,
                        default=TRAIN_IMG, help='directory of training files')
    return parser.parse_args()

if __name__ == '__main__':
    args=get_arguments()
    ###To generate a highres heatmap: do the following:
    sdir='./train_outputs'
    #sdir='./test_outputs'

    with tf.device('/gpu:1'):
        t_fname=args.train_img
        whole_image=get_whole_image(t_fname)
        pat,node=filename_parse(t_fname)
        lr_reader, lr_net = get_lr_network( input_size )
        sess,lr_saver=lr_sess_saver()
        lr_hmap=main( whole_image, lr_net, lr_reader, sess, pat,node,sdir)

        hr_reader, hr_net=get_hr_network(input_size)
        sess,hr_saver=hr_sess_saver(sess,hr_net.variables)

        hr_hmap=hrhm( lr_hmap.img, whole_image, hr_net, hr_reader, sess, pat,node,sdir)

        sess.close()


