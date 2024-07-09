import numpy as np
from skimage import io
import multiresolutionimageinterface as mir
import os

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

def coordinate_change(coord,to):
    '''
    obviously this function doesn't do much,
    and does the same thing regardless of "to"
    but the coordinates were hard to keep track of
    so this is just for the reader
    '''
    assert(to in ['asap','numpy'])
    if to=='asap':
        return np.array([coord[1],coord[0]])
    if to=='numpy':
        return np.array([coord[1],coord[0]])

class OutOfBounds(ValueError):
    pass

class MultiResWrapper(object):
    '''
    This is a wrapper class so that I can program in pure numpy coordinates
    without thinking about how ASAP is weird and transposes them
    '''
    def __init__(self,mr_file):
        reader = mir.MultiResolutionImageReader()
        self.mr_file=mr_file
        self.mr_image= reader.open(self.mr_file)

    def whole_image(self, level):
        ds=self.get_level_downsample(level)
        out_shape=np.round(np.array(self.mr_image.getDimensions())/ds)
        img=self.mr_image.getUCharPatch(0,0,int(out_shape[0]),int(out_shape[1]),level)
        return img

    def extract_patch(self, np_center, np_output_shape, input_level,  output_level):
        '''
        WARN: this method only seems to work safely with levels 0,1, and maybe
        2. I'm using it to sample at level 0 then use np.ndarray[::ds] to
        downsample

        ASAP coordinates - the stuff you read as you curse over /opt/ASAP/bin/ASAP gui
        UCharPatch coordinates - the first two entries of mr_image.getUcharPatch()
        numpy coordinates - the patch[row,col] where patch is returned by UCharPatch

        ASAP coord <--> UCharPatch(level=0)   (ASAP only gives level 0 coords)
        numpy coord <--> transpose( UCharPatch )
            #asking for a 50,100 patch returns a 100,50 numpy array

        #This takes "numpy coordinates" for shape and center

        center is numpy ordered[row,col] tuple and is wrt input_level coordinates
        center is not required to be an integer. Some operations may want to be
        done later, so prevent premature rounding

        output_shape is simply the shape of the output. It is the number of pixels
        at the output level
        '''

        center= coordinate_change(np_center, to='asap')
        output_shape= coordinate_change(np_output_shape, to='asap')

        in_ds=self.get_level_downsample(input_level)
        out_ds=self.get_level_downsample(output_level)

        #convert to level0 coordinates and convert to top left
        top_left = np.round(in_ds*center - out_ds*output_shape/2).astype(np.int)
        bottom_right = np.round(top_left + out_ds*output_shape).astype(np.int)
        mr_shape=self.get_level_dimensions(0)

        if (top_left<0).any() or (bottom_right>=mr_shape).any():
            raise OutOfBounds('sample is out of range of the image')

        #get patch
        patch=self.mr_image.getUCharPatch(top_left[0],top_left[1],output_shape[0],output_shape[1],output_level)

        #convert back to numpy by changing row/col:
        np.swapaxes(patch,axis1=0,axis2=1)
        return patch

    @property
    def spacing(self):
        sp=self.mr_image.getSpacing()
        if len(sp)==0:#some images don't have spacing
            return sp
        else:
            return (sp[1],sp[0])
    @property
    def shape(self):
        return self.get_level_dimensions(0)
    def get_level_dimensions(self, level):
        s=self.mr_image.getLevelDimensions(level)
        return np.array([s[1],s[0]])
    def get_level_downsample(self, level):
        return self.mr_image.getLevelDownsample(level)



def extract_patch(mr_image, np_center, np_output_shape, input_level,  output_level):
    '''

    ASAP coordinates - the stuff you read as you curse over /opt/ASAP/bin/ASAP gui
    UCharPatch coordinates - the first two entries of mr_image.getUcharPatch()
    numpy coordinates - the patch[row,col] where patch is returned by UCharPatch

    ASAP coord <--> UCharPatch(level=0)   (ASAP only gives level 0 coords)
    numpy coord <--> transpose( UCharPatch )
        #asking for a 50,100 patch returns a 100,50 numpy array

    #This takes "numpy coordinates" for shape and center

    center is numpy ordered[row,col] tuple and is wrt input_level coordinates
    center is not required to be an integer. Some operations may want to be
    done later, so prevent premature rounding

    output_shape is simply the shape of the output. It is the number of pixels
    at the output level
    '''

    center= coordinate_change(np_center, to='asap')
    output_shape= coordinate_change(np_output_shape, to='asap')

    in_ds=mr_image.getLevelDownsample(input_level)
    out_ds=mr_image.getLevelDownsample(output_level)

    #convert to level0 coordinates and convert to top left
    top_left = np.round(in_ds*center - out_ds*output_shape/2).astype(np.int)

    #get patch
    patch=mr_image.getUCharPatch(top_left[0],top_left[1],output_shape[0],output_shape[1],output_level)

    #convert back to numpy by changing row/col:
    np.swapaxes(patch,axis1=0,axis2=1)
    return patch

#scratch:::
#                try:
#                    #we were getting error:
#                    #OverflowError: in method MultiResolutionImage_getUCharPatch,
#                    #argument 4 of type unsigned long long
#                    image_patch=mr_image.getUCharPatch(0,0,np.int(n_rows/ds),np.int(n_cols/ds),ds_level)
#                except Exception as ex:
#                    print type(ex)
#                    print ex.args
#                    print ex
#                    continue
#    if display_samples:
#        copy = np.copy(image_patch)
#        for x,y in center_pixels:
#            copy[x,y,:] = [255,0,0]
#        io.imshow(copy)
#        io.show()
#
