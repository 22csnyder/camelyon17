import openslide
from openslide import OpenSlideError
import numpy as np
import multiresolutionimageinterface as mir
from skimage import io
import os
import glob
from utils import extract_patch,filename_parse


#final test:
#see if downsample and get image commute (!)
if __name__=='__main__':
    #Filename then ASAP coords of some region of interest given:

    #fn='/mnt/md0/test_data/centres/patient_145_node_2.tif'
    #top_left,bot_right=np.array([71838,94150]),np.array([72411,94613])#145_2

    #fn='/mnt/md0/CAMELYON_2017/centre_4/patient_092_node_3.tif'
    #top_left,bot_right=np.array([20700,40400]), np.array([22400,42100])

    #fn='/mnt/md0/CAMELYON_2017/centre_2/patient_041_node_0.tif'
    #top_left,bot_right=np.array([83874,26091]) , np.array([85218,28435])#41_0

    fn='/mnt/md0/CAMELYON_2017/centre_4/patient_081_node_4.tif'
    top_left,bot_right=np.array([65500,20500]) , np.array([71000,24000])#81_4

    sdir='/home/chris/Projects/camelyon17/figures'
    pat,node=filename_parse(fn)
    #savef='figures/patient_'+str(pat)+'_'+str(node)+'_'+str(top_left[0])+'_'+str(top_left[1])
    save_info=['patient',pat,'node',node,top_left[0],top_left[1]]
    fmat_info='_'.join(map(str,save_info))
    savef='figures/'+fmat_info

    shape0=bot_right-top_left
    shape3=np.round( shape0.astype('float')/8).astype('int')

    #mir_image
    reader = mir.MultiResolutionImageReader()
    mr_image= reader.open(fn)
    lvl0=0; lvl3=3
    ds0=int(mr_image.getLevelDownsample(lvl0))
    ds3=int(mr_image.getLevelDownsample(lvl3))
    sz0=mr_image.getLevelDownsample(lvl0)
    sz3=mr_image.getLevelDownsample(lvl3)
    out_shape0=np.round(np.array(mr_image.getDimensions())/ds0)
    out_shape3=np.round(np.array(mr_image.getDimensions())/ds3)
    whole_shape3=mr_image.getLevelDimensions(lvl3)

    mI0=mr_image.getUCharPatch(top_left[0],top_left[1],shape0[0],shape0[1],lvl0)
    mI3=mr_image.getUCharPatch(top_left[0],top_left[1],shape3[0],shape3[1],lvl3)
    whole_mI3=mr_image.getUCharPatch(0,0,whole_shape3[0],whole_shape3[1],lvl3)

    io.imsave(savef+'_mrlvl0.tif',mI0[::ds3,::ds3])
    io.imsave(savef+'_mrlvl3.tif',mI3)
    io.imsave(savef+'_wholeds20_mrlvl3.tif',whole_mI3[::20,::20])

    ##os_image
    os_image=openslide.open_slide(fn)
    ds0=os_image.level_downsamples[lvl0]
    ds3=os_image.level_downsamples[lvl3]
    sz0=os_image.level_dimensions[lvl0]
    sz3=os_image.level_dimensions[lvl3]

    ds0=np.int(ds0)
    ds3=np.int(ds3)

    #img0=os_image.read_region(top_left, lvl0, np.transpose(shape0))
    try:
        img0=os_image.read_region(top_left, lvl0, shape0)
        img3=os_image.read_region(top_left, lvl3, shape3)
        whole_img3=os_image.read_region((0,0), lvl3, sz3)

        Img0=np.array(img0)
        Img3=np.array(img3)
        WImg3=np.array(whole_img3)
        io.imsave(savef+'_oslvl0.tif',Img0[::ds3,::ds3])
        io.imsave(savef+'_oslvl3.tif',Img3)
        io.imsave(savef+'_wholeds20_oslvl3.tif',WImg3[::20,::20])

    except OpenSlideError as ose:
        print type(ose)
        print ose



##Can we access each of the files: -> no:21_0, 21_1
#    train_files=glob.glob('/mnt/md0/CAMELYON_2017/centre*/*.tif')
#    test_dir='/mnt/md0/test_data/centres'
#    test_files=sorted( glob.glob(test_dir+'/*.tif'),key=lambda f: filename_parse(f))
#
#    for f in test_files:
#        try:
#            osi=openslide.open_slide(f)
#        except:
#            print 'error:',os.path.basename(f)



#    test_dir='/mnt/md0/test_data/centres'
#    test_files=sorted( glob.glob(test_dir+'/*.tif'),key=lambda f: filename_parse(f))
#    ##March 31 try again. new strategy: fix openslide
#    t_fname=test_files[105]
#
#
#    lvl=3
#    os_image=openslide.open_slide(t_fname)
#    sz=os_image.level_dimensions[lvl]
#    ds=os_image.level_downsamples[lvl]
#
#    os_image.read_region((0,0),ds,sz)



#_--------------------------------
#reader = mir.MultiResolutionImageReader()
#
#fname_20_04='/home/chris/Data/CAMELYON_2017/centre_1/patient_020/patient_020_node_4.tif'
#mr_image_20_04 = reader.open(fname_20_04)
#
#
#test_files=sorted( glob.glob(test_dir+'/*.tif'),key=lambda f: filename_parse(f))
#
#
#
##fname='/mnt/sdb1/Data/CAMELYON_2017/centre_4/patient_099_node_4.tif'
#
#    t0=time.time()
#    reader = mir.MultiResolutionImageReader()
#    mr_image= reader.open(image_file)
#    ds=mr_image.getLevelDownsample(lvl)
#    out_shape=np.round(np.array(mr_image.getDimensions())/ds)
#    whole_image=mr_image.getUCharPatch(0,0,int(out_shape[0]),int(out_shape[1]),lvl)
#    print 'lvl',lvl,' image read complete (',(time.time()-t0)/60.,'min)'
#    return whole_image

###As of Friday, seems to work rather well:
#reader = mir.MultiResolutionImageReader()
#
#fname='/mnt/sdb1/Data/CAMELYON_2017/centre_4/patient_081_node_4.tif'
#fmask='/mnt/sdb1/Data/CAMELYON_2017/lesion_masks/patient_081_node_4_mask.tif'
#
#
#mr_image = reader.open(fname)
#mr_mask = reader.open(fmask)
#
#
##lesion in p81_n4
##ASAP coordinates:
#top_left=np.array([66380,20888])
#bot_right=np.array([66853,21310])
#shape=bot_right-top_left
#
#shape=bot_right-top_left
#savef='figures/patient_81_node_4_'+str(top_left[0])+'_'+str(top_left[1])
#savemask=savef+'_mask.tif'
#savefile=savef+'.tif'
#
#
#mask_patch=mr_mask.getUCharPatch(top_left[0],top_left[1],shape[0],shape[1],0)
#image_patch=mr_image.getUCharPatch(top_left[0],top_left[1],shape[0],shape[1],0)
###Equivalent:#
##X=mr_image.getUCharPatch(0,0,imshape[0],imshape[1],0)
##np_patch=X[top_left[1]:bot_right[1],top_left[0]:bot_right[0]]
###np_patch == image_patch elementwise
#
#io.imsave(savefile,image_patch)
#io.imsave(savemask,mask_patch)
#
#
#center=(top_left+bot_right)/2
#np_center= np.array([center[1],center[0]])
#np_shape=np.array([shape[1],shape[0]])
#
#c_patch=extract_patch(mr_image, np_center,np_shape,0,0)
#io.imsave(savef+'_center_patch_method.tif',c_patch)




#level = 2
#ds = mr_image.getLevelDownsample(level)  #0.2 ms
#
#
##level, time:
## 0   , 6ms
## 1   , 3ms
## 2   , 1.5 ms
## 3   , 5ms
#
##shape: (200,300,3)
#
##Experiements show:
#    #topleft row, topleft col, n_rows_output, n_col_output, downsamplelevel
#image_patch = mr_image.getUCharPatch(int(568 * ds), int(732*ds),300,200,level)
#
#
#
##batch~256.. 7ms*256= 1.7s per batch  # ouch
#
#
#import multiresolutionimageinterface as mir
#reader = mir.MultiResolutionImageReader()
#mr_image = reader.open(fname)
#annotation_list = mir.AnnotationList()
#xml_repository = mir.XmlRepository(annotation_list)
#
#anno_source='/home/chris/Data/Camelyon/lesion_annotations/patient_020_node_4.xml'
#xml_repository.setSource(anno_source)
#xml_repository.load()
#annotation_mask = mir.AnnotationToMask()
#
#label_map = {'metastases': 1, 'normal': 2}
#
#output_path='/home/chris/Data/Camelyon/scratch/anno'
#annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map)









