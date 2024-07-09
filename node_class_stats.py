import numpy as np
from skimage import io
import glob
import multiresolutionimageinterface as mir
from utils import filename_parse
from scipy.ndimage.filters import maximum_filter
from skimage.measure import regionprops, label
import cPickle as pickle

GT_FILE='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/stage_labels.csv'
HEATMAP_DIR='/home/chris/Projects/camelyon17/tensorflow-deeplab-lfov/train_outputs/heatmaps/*.tif'
MASK_DIR = '/mnt/md0/CAMELYON_2017/lesion_masks/*.tif'

def get_whole_image(image_file,lvl=3):
    reader = mir.MultiResolutionImageReader()
    mr_image= reader.open(image_file)
    if mr_image is None: raise ValueError('no file present:')
    ds=mr_image.getLevelDownsample(lvl)
    if ds == -1: raise ValueError('negative ds:')
    out_shape=np.round(np.array(mr_image.getDimensions())/ds)
    whole_image=mr_image.getUCharPatch(0,0,int(out_shape[0]),int(out_shape[1]),lvl)
    return whole_image

def largest_conn_region(regions):
    max_region = 0.0
    max_idx = 0
    max_label = 0
    for idx in range(len(regions)):
        if regions[idx]['area'] > max_region and regions[idx]['label']!=0:
            max_region = regions[idx]['area']
            max_label = regions[idx]['label']
            max_idx = idx
    area = regions[max_idx]['area']
    maj_ax = regions[max_idx]['major_axis_length']
    perimeter = regions[max_idx]['perimeter']
    return area,maj_ax,perimeter

if __name__ == '__main__':
    threshold=0.2
    class_dict = {}
    lines = [line.rstrip('\r\n') for line in open(GT_FILE)]

    for filename in glob.iglob(MASK_DIR):
        try:
            mask = get_whole_image(filename)
            mask = mask.reshape(mask.shape[0],mask.shape[1])
            if np.mean(mask) > 0.9: continue
        except Exception as error:
            print(repr(error) + ' ' + filename)
            continue
        elements= filename.split('/')[-1].split('_')
        fn='patient_'+elements[1]+'_node_'+elements[3]+'.tif'
        for line in lines:
            f,lab = line.split(',')
            if fn==f:
                conn_regs, num = label(mask,background=0,return_num=True,connectivity=2)
                regions = regionprops(conn_regs,mask)
                area,maj_ax,perimeter = largest_conn_region(regions)
                if lab not in class_dict:
                    class_dict[lab]={'area':[area],'maj_ax':[maj_ax],'per':[perimeter]}
                else:
                    class_dict[lab]['area'].append(area)
                    class_dict[lab]['maj_ax'].append(maj_ax)
                    class_dict[lab]['per'].append(perimeter)
    with open('/home/chris/Projects/camelyon17/class_stats.pkl','w+') as f:
        pickle.dump(class_dict,f)

