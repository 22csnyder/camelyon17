import openslide
from openslide import OpenSlideError
from skimage import io, exposure
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import img_as_uint
from os import listdir
from os.path import isfile, join, basename
import time
import os, sys
from openslide import OpenSlide
from utils import MultiResWrapper,filename_parse

import warnings
import argparse

ROOT_DIR = '/mnt/md0/CAMELYON_2017'
MASK_DIR = '/mnt/md0/CAMELYON_2017/lesion_masks'
OUT_DIR = '/mnt/md0/CAMELYON_2017/training_samples'


#TODO: add help methods
#      clean up unused arguments
def get_arguments():
    parser=argparse.ArgumentParser(description='takes a data directory and looks\
                                   for tif/mask file combos to generate a glimpses\
                                  of larger images to be used as training examples')

    parser.add_argument('--rootdir',type=str,default=ROOT_DIR,
                       help='main directory containing all tif and mask files.\
                        Program assumes all tif files within are either \
                        mask, input, or output files')
    parser.add_argument('--maskdir',type=str,default=MASK_DIR,help='directory containing mask files that match tif filenames')
    parser.add_argument('--outputdir',type=str,default=OUT_DIR)
    parser.add_argument('--samples_per_image',type=int,default=300)
    parser.add_argument('--ds_level',type=int,default=5)
    parser.add_argument('--sl_level',type=int,default=3)
    parser.add_argument('--threshold',type=int,default=0.025)
    parser.add_argument('--border_multiplier',type=int,default=6)
    parser.add_argument('--seed',type=int, default=7654)
    return parser.parse_args()


#first whitens black pixels then performs otsu threshold
def otsu_threshold(image):
    gray_img = rgb2gray(image)
    # whiten black pixels
    black = np.where(gray_img < 0.1)
    gray_img[black] = 1.0
    # threshold to get cells
    try:
        thresh = threshold_otsu(gray_img)
        return gray_img > thresh
    except:
        return None

def sample_window(mr_image, pixel, window_size, sl_level):
    x, y = pixel
    if(y-window_size/2 < 0) or (x-window_size/2 < 0):
        return None
    else:
        sample_300x300 = mr_image.getUCharPatch(y-window_size/2,
                                                x-window_size/2,
                                                window_size,window_size,
                                                sl_level)
        return sample_300x300


def mkdir(dire):
    if not os.path.exists(dire):
        print 'making new directory:',dire
        os.makedirs(dire)

def sliding_window(input_file,
                   output_dir,
                   stride=300,
                   sample_dim = 300,
                   no_skip=False,
                   sl_level=0
                   ):
    '''
    This function takes a multiresolution tif file and outputs a directory
    of sliding window files that tile that image

    It skips very dark and very light images

    '''
    if not os.path.exists(input_file):
        raise ValueError('file %s does not exist'% input_file)


    base_name_dot_tif=os.path.basename(input_file)
    base_name,ftype=base_name_dot_tif.split('.')
    assert ftype in ['tif','tiff']
    pt_window_dir=os.path.join(output_dir,base_name)
    mkdir(pt_window_dir)
    _,patient, _,node= base_name.split('_')

    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(input_file)
    n_rows, n_cols = mr_image.getDimensions()
    ds = mr_image.getLevelDownsample(sl_level)

    n_skipped_dark,n_skipped_light=0,0
    print('Sampling from input_file: %s' % input_file)
    for i in np.arange(0,int(n_cols/ds),stride):
        print 'col ',i,' outof ',int(n_cols/ds)
        for j in np.arange(0,int(n_rows/ds),stride):
            sample_300x300 = mr_image.getUCharPatch(int(j*ds),int(i*ds),sample_dim,sample_dim,sl_level)
            #gray_300x300 = rgb2gray(sample_300x300)
            #mean = np.mean(gray_300x300)
            if not no_skip:
                if mean <0.1*256:
                    n_skipped_dark+=1
                    continue
                elif mean>0.97*256:
                    n_skipped_light+=1
                    continue
            print('saving sample')
            fname=os.path.join(pt_window_dir,'patient_%s_%s_%s_%s.tif' % (patient,node,j,i) )
            #fname=os.path.join(pt_window_dir,'patient_%s_%s_%s_%s.jpg' % (patient,node,j,i) )
            io.imsave(fname, sample_300x300)

    print 'skipped ',n_skipped_light,' windows due to white'
    print 'skipped ',n_skipped_dark,' windows due to black'

def rand_subset(L,size):
    #Choose randomly up to "size" unique elements
    np_samples=np.random.permutation(L)[:size]
    #uniquify by returning set
    return {tuple(pix) for pix in np_samples}


def morphological_sample(image,mask,samples_per_image,patient_num,node,border_multiplier):
    binary = otsu_threshold(image)
    #If whole image can't be thresholded (almost nvr happens)
    if binary is None:
        #patient_96_node_0
        raise ValueError('rejecting image(!) for sparsity')
    cell_pixels = np.where(binary==False)
    cell_pixels = zip(cell_pixels[0], cell_pixels[1])

    ds_dilated = binary_dilation(mask,selem=np.ones((3,3)))
    dilated_pixels = np.where(ds_dilated==1.0)
    dilated_pixels = zip(dilated_pixels[0],dilated_pixels[1])
    dilated_pixels = set(dilated_pixels)

    tumor_pixels = np.where(mask==1.0)
    tumor_pixels = zip(tumor_pixels[0],tumor_pixels[1])
    tumor_pixels = set(tumor_pixels)

    border_pixels = dilated_pixels.difference(tumor_pixels)
    border_pixels = list(border_pixels)
    cell_pixels = set(cell_pixels)
    normal_pixels = cell_pixels.difference(tumor_pixels)
    tumor_pixels = list(tumor_pixels)
    normal_pixels = list(normal_pixels)
    if len(border_pixels)==0:
        print 'WARN:border method fails:patient:',patient_num,' node:',node
    sz_normal=samples_per_image
    sz_tumor=samples_per_image
    sz_border=samples_per_image*border_multiplier

    center_pixels=np.vstack(set.union(
            rand_subset( normal_pixels, sz_normal ),
            rand_subset( tumor_pixels,  sz_tumor  ),
            rand_subset( border_pixels, sz_border )
            ))

    n_samples=len(center_pixels)
    return center_pixels

def heatmap_sample(image,mask,samples_per_image,threshold):
    high = (image > threshold)
    pos_pixels = np.where(high==1.0)
    pos_pixels = zip(pos_pixels[0],pos_pixels[1])
    center_pixels = rand_subset(pos_pixels,samples_per_image)
    return center_pixels

############################
# generate_training_samples
# generates 300x300 samples from image at highest resolution (level=0)
# rootdir = where your images are e.g. /home/kenleejr92/CAMELYON_2017
# samples_per_image = how many pixels to sample per .tif
# Note: each pixel sampled will then generate many training images from sliding window
# ds_level = down sample level at which thresholding and random sampling of pixels is performed
# window_size = once a random pixel is sampled, a window surrounding the pixel is retrieved at highest resolution
#               a sliding window of size 300x300 is then passed over the window and saved along with ground truth
# stride = stride of sliding window
#(y*ds,x*ds) is index of random pixel at lowest resolution
###########################
def generate_training_samples(rootdir='/home/kenleejr92/Desktop',
                              maskdir='/home/kenleejr92/Desktop/masks',
                              outputdir='/home/kenleejr92/Desktop/CAMELYON_2017/training_samples',
                              samples_per_image=300,
                              ds_level=4,
                              sl_level=3,
                              sampling_type='morphological',
                              threshold=0.025,
                              network_type='include',
                              border_multiplier=6,
                              seed=1234):
    '''
    dont put a slash on the end of directories you pass
    sampling_type is morphological or heatmap
    text file holding where samples are in [network_type]_[sampling_level]_dataset.txt
    '''
    if len(maskdir)==0:
        maskdir=rootdir+'/lesion_masks'

    assert(rootdir[-1]!='/')
    assert(maskdir[-1]!='/')
    assert(outputdir[-1]!='/')
    #assert(xs_level in [0,1])
    #assert(xs_level in [0])#only way to go
    patch_shape=[samples_per_image,samples_per_image]

    assert(ds_level>=sl_level)

    mkdir(rootdir)
    mkdir(maskdir)
    mkdir(outputdir)
    text_file_dir=outputdir+'/text_files'
    mkdir(text_file_dir)

    np.random.seed(seed)
    sample_dim = 300
    patch_shape=np.array([sample_dim,sample_dim])
    default_empty_mask = np.zeros((300,300), np.int8)

    #tar_dir='/CAMELYON_%s/training_samples' % year
    #remove slash at end if exists:
    tar_dir=outputdir.split('/')[-1]#just get last component
    dataset_id = network_type + '_' + str(sl_level) + '_'
    f1= open(os.path.join(text_file_dir,dataset_id+'dataset.txt'), 'a')#append
    i_sample=0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            #Checks for all files ending with ".tif" not maskdir or outputdir
            if filepath.endswith(".tif") and subdir != outputdir and subdir != maskdir:
                if i_sample>0:
                    #not ideal as will not trigger at end
                    print '..created ',i_sample, 'image patches'
                    i_sample=0
                patient_num,node=filename_parse(file)
                imagedir = outputdir + '/patient_%s_node_%s' % (patient_num, node)
                tartxt_dir = tar_dir + '/patient_%s_node_%s' % (patient_num, node)
                pat = '%03d' % patient_num
                node=str(node)
                #Check if has a mask file
                fmask=maskdir+'/patient_%s_node_%s_mask.tif' % (pat, node)
                os_masks = None
                if os.path.isfile(fmask):
                    #ground_truth = reader.open(maskdir+'/patient_%s_node_%s_mask.tif' % (patient_num, node))
                    os_mask = openslide.open_slide(fmask)
                    has_ground_truth = True
                    print 'Sampling from file: %s' % file
                else:
                    has_ground_truth = False
                    print 'no ground truth for %s_node_%s' % (patient_num,node),
                    print '..ignoring'
                    continue
                ##past here assumes mrw_mask exists

                mkdir(imagedir)
                ##os_image
                os_image=openslide.open_slide(filepath)
                ds=os_image.level_downsamples[ds_level]
                sl=os_image.level_downsamples[sl_level]
                sz_ds=os_image.level_dimensions[ds_level]
                sz_sl=os_image.level_dimensions[sl_level]
                ds=np.int(ds)
                sl=np.int(sl)

                print 'loading data at ds=',ds,', patient %s node %s to perform morphology'% (patient_num, node),
                t0=time.time()
                ds_image=np.array(os_image.read_region((0,0),ds_level,sz_ds))[:,:,:3]
                ds_mask=np.array(os_mask.read_region((0,0),ds_level,sz_ds))[:,:,0]#[...,0]
                ds_mask=np.squeeze(ds_mask)
                print '..total time',time.time()-t0,'(s)'

                if sampling_type == 'morphological':
                    try:
                        ds_center_pixels =morphological_sample(ds_image,ds_mask,samples_per_image,patient_num,node,border_multiplier)
                    except ValueError as err:
                        print repr(err)
                        continue
                elif sampling_type == 'heatmap':
                    ds_center_pixels=heatmap_sample(ds_image,samples_per_image,threshold)
                #ds to sl coordinates
                sl_center_pixels=ds_center_pixels*ds#/sl*sl
                #there is some unknown as to what position was being referenced
                #also in effect this adds in a little bit of noise
                n_samples = len(sl_center_pixels)
                #sl_location_uncertianty=np.random.randint(0,ds/sl,2*n_samples).reshape((n_samples,2))
                sl_location_uncertianty=np.random.randint(0,ds,2*n_samples).reshape((n_samples,2))
                sl_center_pixels+=sl_location_uncertianty

                patch_shape=np.array(patch_shape)
                sl_patch_shape=sl*patch_shape
                sl_top_left_pixels=np.round(sl_center_pixels-sl_patch_shape/2).astype(np.int)
                #sl_bot_right_pixels=sl_top_left_pixels+patch_shape

                #for tl,br in zip(sl_top_left_pixels,sl_bot_right_pixels):
                for tl in sl_top_left_pixels:
                    patch_image=np.array(os_image.read_region(tl[::-1],sl_level,patch_shape))[:,:,:3]
                    patch_mask=np.array(os_mask.read_region(tl[::-1],sl_level,patch_shape))[:,:,0]
                    binary = otsu_threshold(patch_image)
                    if binary is None:
                        #print('rejecting sample for sparsity')
                        if np.random.rand()<0.01:
                            continue
                    if np.mean(binary) >=0.97:
                        if np.random.rand()<0.01:
                            continue
                        #print('rejecting sample for sparsity')

                    i_sample+=1
                    #print('Creating sample around center pixel: (%s,%s)'%(int(ds*r),int(ds*c)))

                    savef='/patient_%s_%s_%s_%s_%s' % (patient_num,node,tl[1]*sl,tl[0]*sl,sl)
                    indicator = str(np.max(patch_mask))
                    #mask saving generates too many low contrast warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        io.imsave(imagedir+savef+'.jpg',patch_image)
                        io.imsave(imagedir+savef+'.png', patch_mask)
                    f1.write(tartxt_dir+savef+'.jpg'+tartxt_dir+savef+'.png'+indicator+'\n')
    f1.close()#finish files loop
    if i_sample>0:
        print '..created ',i_sample, 'image patches'

if __name__ == '__main__':
     args=get_arguments()
     kwargs= vars(args)
     generate_training_samples( **kwargs)



