import numpy as np
from skimage import io
import glob
import multiresolutionimageinterface as mir
from utils import filename_parse
from scipy.ndimage.filters import maximum_filter
from math import sqrt
import warnings

GT_FILE='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/stage_labels.csv'
HEATMAP_DIR='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/train_outputs/heatmaps/*.tif'
MASK_DIR = '/mnt/md0/CAMELYON_2017/lesion_masks'

def get_whole_image(image_file,lvl=3):
    reader = mir.MultiResolutionImageReader()
    mr_image= reader.open(image_file)
    if mr_image is None:raise ValueError('image not found')
    ds=mr_image.getLevelDownsample(lvl)
    if ds == -1: raise ValueError('negative ds')
    out_shape=np.round(np.array(mr_image.getDimensions())/ds)
    whole_image=mr_image.getUCharPatch(0,0,int(out_shape[0]),int(out_shape[1]),lvl)
    return whole_image

def closest_point(point,plist):
    x1,y1=point
    min_dist = 5000
    closest_point = (0,0)
    for x2,y2 in plist:
        dist = sqrt((x1-x2)**2+(y1-y2)**2)
        if dist <= min_dist:
            min_dist = dist
            closest_point = (x2,y2)
    return closest_point

if __name__ == '__main__':
    threshold=0.025
    itcs=[]
    reader = mir.MultiResolutionImageReader()
    lines = [line.rstrip('\r\n') for line in open(GT_FILE)]
    for line in lines:
        filename,label = line.split(',')
        if label=='itc':
            itcs.append(filename)
    for filename in glob.iglob(HEATMAP_DIR):
        for itc in itcs:
            if filename.split('/')[-1].split('_')[0]=='lowres':
                pat,node =filename_parse(itc)
                pat2,node2=filename_parse(filename)
                if pat==pat2 and node==node2:
                    pat = '%03d' % pat
                    node=str(node)
                    img = io.imread(filename)
                    try:
                        mask = get_whole_image(MASK_DIR+'/patient_'+pat+'_node_'+node+'_mask.tif',lvl=3)
                    except Exception as error:
                        print(repr(error) + ' %s' % MASK_DIR+'/patient_'+pat+'_node_'+node+'_mask.tif')
                        continue
                    mask = mask.reshape(mask.shape[0],mask.shape[1])
                    pos = np.array(img>threshold)
                    filter_mask = np.ones((300,300))
                    mask = maximum_filter(mask,footprint=filter_mask,mode='constant',cval=0)
                    mask = mask[::300,::300]
                    mask = mask[:pos.shape[0],:pos.shape[1]]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        io.imsave('/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/train_outputs/heatmaps/mask_'+pat+'_'+node+'.tif',mask)
                    print('patient: ' + pat + ' node: ' + node)
                    P = np.where(mask==1)
                    P = zip(P[0],P[1])
                    C = np.where(pos==1)
                    C = zip(C[0],C[1])
                    print('mask positives at:')
                    print(P)
                    print('closest include positives to mask positives')
                    for x,y in P:
                        cp = closest_point((x,y),C)
                        if cp[0]==0 and cp[1]==0:
                            print('could not find itc')
                        else:
                            print(cp)
                    mask_pos = (mask==1)
                    hm_pos = (pos==1)
                    TP = np.sum(np.multiply(mask_pos,hm_pos))
                    FNR = 1-TP/np.sum(mask_pos)


