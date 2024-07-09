import numpy as np
import multiresolutionimageinterface as mir
from skimage import io
from skimage.morphology import binary_dilation
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import img_as_uint
from skimage.filters import threshold_otsu

from utils import extract_patch
from utils import MultiResWrapper,OutOfBounds


import time
from generate_training_samples import threshold
import multiresolutionimageinterface as mir



fname='/mnt/sdb1/Data/CAMELYON_2017/centre_4/patient_081_node_4.tif'
fmask='/mnt/sdb1/Data/CAMELYON_2017/lesion_masks/patient_081_node_4_mask.tif'

samples_per_image=1000
border_multiplier=1

#xs_level either 0 or 1
xs_level=0
xs=2**xs_level
print 'xs_level is ',xs_level

ds_level=4#level to do mask
ds=2**ds_level

sampling_level=3
sl=2**sampling_level

assert(ds>=sl>=xs)
patient_num=81
patch_shape=[300,300]
node=4

mrw_image = MultiResWrapper(fname)
mrw_mask = MultiResWrapper(fmask)

T=[]
T.append(time.time())

#This code assumes getLevelDownsample(1)=2
xs_image=mrw_image.whole_image(xs_level)
print 'finish reading xs_image'

xs_mask=np.squeeze(mrw_mask.whole_image(xs_level))
print 'finish reading mask'
T.append(time.time())

ds_image=xs_image[::ds/xs,::ds/xs]
sl_image=xs_image[::sl/xs,::sl/xs]
ds_mask=xs_mask[::ds/xs,::ds/xs]
sl_mask=xs_mask[::sl/xs,::sl/xs]
T.append(time.time())

###################random sampling of pixels##################
binary = threshold(ds_image)
cell_pixels = np.where(binary==False)
cell_pixels = zip(cell_pixels[0], cell_pixels[1])
T.append(time.time())

ds_dilated = binary_dilation(ds_mask,selem=np.ones((3,3)))#
T.append(time.time())

dilated_pixels = np.where(ds_dilated==1.0)
dilated_pixels = zip(dilated_pixels[0],dilated_pixels[1])
dilated_pixels = set(dilated_pixels)
T.append(time.time())
tumor_pixels = np.where(ds_mask==1.0)
tumor_pixels = zip(tumor_pixels[0],tumor_pixels[1])
tumor_pixels = set(tumor_pixels)
border_pixels = dilated_pixels.difference(tumor_pixels)
border_pixels = list(border_pixels)
cell_pixels = set(cell_pixels)
normal_pixels = cell_pixels.difference(tumor_pixels)
tumor_pixels = list(tumor_pixels)
normal_pixels = list(normal_pixels)
T.append(time.time())

nlp=normal_pixels#DEBUG

if len(border_pixels)==0:
    print 'border method fails:patient:',patient_num,' node:',node
    #continue

sz_normal=samples_per_image
sz_tumor=samples_per_image
sz_border=samples_per_image*border_multiplier

def rand_subset(L,size):
    #Choose randomly up to "size" unique elements
    np_samples=np.random.permutation(L)[:size]
    #uniquify by returning set
    return {tuple(pix) for pix in np_samples}

ds_center_pixels=np.vstack(set.union(
    rand_subset( normal_pixels, sz_normal ),
    rand_subset( tumor_pixels,  sz_tumor  ),
    rand_subset( border_pixels, sz_border )
    ))
n_samples=len(ds_center_pixels)

#ds to sl coordinates
sl_center_pixels=ds_center_pixels*ds/sl#should still be int #xs cancels

#there is some unknown as to what position was being referenced
#also in effect this adds in a little bit of noise
sl_location_uncertianty=np.random.randint(0,ds/sl,2*n_samples).reshape((n_samples,2))
sl_center_pixels+=sl_location_uncertianty
T.append(time.time())

#sl_center=sl_center_pixels[0]

patch_shape=np.array(patch_shape)
sl_top_left_pixels=np.round(sl_center_pixels-patch_shape/2).astype(np.int)
sl_bot_right_pixels=sl_top_left_pixels+patch_shape

i=0
for tl,br in zip(sl_top_left_pixels[:5],sl_bot_right_pixels[:5]):
    patch_image=sl_image[tl[0]:br[0],tl[1]:br[1]]
    patch_mask=sl_mask[tl[0]:br[0],tl[1]:br[1]]

    savef='figures/patient_81_node_4_'+str(tl[0])+'_'+str(tl[1])
    io.imsave(savef+'_borderpatchv2_img'+str(i)+'_xs'+str(xs_level)+'.tif',patch_image)
    io.imsave(savef+'_borderpatchv2_mask'+str(i)+'_xs'+str(xs_level)+'.tif',patch_mask)
    T.append(time.time())

    i+=1

def showtime(T):
    for i in range(len(T)-1):
        print 't'+str(i)+':',T[i+1]-T[i]
showtime(T)
#io.imsave(savef+'_borderpatchv2_img_xs'+str(xs_level)+'.tif',patch_image)
#io.imsave(savef+'_borderpatchv2_mask_xs'+str(xs_level)+'.tif',patch_mask)

