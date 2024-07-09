import os

import numpy as np
import tensorflow as tf

#IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

#def read_image_directory(data_dir):
#    """
#    This is to be used when there are no labels to speak of
#    therefore it is only used during inference
#    Reads and returns all images in directory
#
#    Args:
#      data_dir: path to the directory with images and masks.
#
#    Returns:
#      Two lists with all file names for images and masks, respectively.
#    """
#    f = open(data_list, 'r')
#    images = []
#    for line in f:
#        image, mask = line.strip("\n").split(' ')
#        images.append(data_dir + image)
#        masks.append(data_dir + mask)
#    return images, masks

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(data_dir + '/' + image)
        masks.append(data_dir + '/' + mask)
    return images, masks

def read_unlabeled_images_from_disk(input_queue, input_size, random_scale):
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.

    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_filename = input_queue[0]
    img_contents = tf.read_file(input_queue[0])
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_filename = tf.convert_to_tensor(img_filename, dtype=tf.string)
    if input_size is not None:
        h, w = input_size
        if random_scale:
            scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
            h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
            w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
            new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
            img = tf.image.resize_images(img, new_shape)
        img = tf.image.resize_image_with_crop_or_pad(img, h, w)
    # RGB -> BGR.
    img_r, img_g, img_b = tf.split(img, 3, axis=2)
    #img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img = tf.cast(tf.concat([img_b, img_g, img_r],2 ), dtype=tf.float32)
    # Extract mean.
    #img -= IMG_MEAN # do preprocessing separately
    return img, img_filename

def format_image(img,label,input_size):
    '''
    img: by intention this is the image as it is read from disk
    label: likewise. If no label (inference), pass None
    input_size: The shape of the input to the tf model.
        not the shape of the raw input
    '''
    if input_size is not None:
        h,w=input_size
        img = tf.image.resize_image_with_crop_or_pad(img, h, w)
        if label is not None:
            label = tf.image.resize_image_with_crop_or_pad(label, h, w)

    # RGB -> BGR.
    img_r, img_g, img_b = tf.split(img, 3, axis=2)
    #img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img = tf.cast(tf.concat([img_b, img_g, img_r],2 ), dtype=tf.float32)
    # Extract mean.
    #img -= IMG_MEAN # do preprocessing separately
    return img, label

#TODO
#This is a workaround until I build tensorflow r1.1 from source
#in r1.0 tf.image.resize_image_with_crop_or_pad only takes 3D tensors (not batch)
#this is fixed in r1.1
#the cpu will have to make sure images are the correct size
def format_image_without_resize(img,label,input_size):
    '''
    img: by intention this is the image as it is read from disk
    label: likewise. If no label (inference), pass None
    input_size: The shape of the input to the tf model.
        not the shape of the raw input
    '''
    #ignore input_size

    # RGB -> BGR.
    img_r, img_g, img_b = tf.split(img, 3, axis=3)#note chanel_axis=3
    img = tf.cast(tf.concat([img_b, img_g, img_r],3 ), dtype=tf.float32)#note chanel_axis=3
    return img, label


def read_images_from_disk(input_queue, input_size, random_scale):
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.

    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_filename = input_queue[0]
    label_filename = input_queue[1]
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    img = tf.image.decode_jpeg(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)
    img_filename = tf.convert_to_tensor(img_filename, dtype=tf.string)
    label_filename = tf.convert_to_tensor(label_filename, dtype=tf.string)
    if input_size is not None:
        h, w = input_size
        if random_scale:
            scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
            h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
            w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
            new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
            img = tf.image.resize_images(img, new_shape)
            label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
            label = tf.squeeze(label, squeeze_dims=[0]) # resize_image_with_crop_or_pad accepts 3D-tensor.
    img,label=format_image(img,label,input_size)

    return img, label, img_filename, label_filename


class ImageInput(object):
    #for placeholder
    #by intention this will be used for inference
    def __init__(self, ph_size,input_size=None,random_scale=False,dtype=tf.uint8):

        ph_shape=[None]+list(ph_size)+[3]
        input_shape=[None]+list(input_size)+[3]
        label_shape=[None]+list(input_size)

        self.ph_image=tf.placeholder(dtype, ph_shape, name='ph_image')
        #self.ph_label=tf.placeholder(dtype,[-1]+list(input_size),name='ph_label')

        #TODO:
        #if random: do_random_image_noise()

        #TODO:upgrade tensorflow to enable batch resizing
        #rescale, to_float, to_bgr
        #self.image, _=format_image(self.ph_image, label=None, input_size)
        self.image_batch,_=format_image_without_resize(self.ph_image, label=None,input_size=input_size)


class ImageReader(object):
    #TODO perhaps would go faster if queue held images instead of filenames...
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir=None, data_list=None, input_size=None, random_scale=None,
                 coord=None,inference_dir=None,num_epochs=None,shuffle=True):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          coord: TensorFlow queue coordinator.
          inference_dir(optional): pass a directory of images to do inference on
          num_epochs: pass only if you want to go for num_epochs then stop
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        self.inference_dir=inference_dir
        self.num_epochs=num_epochs
        self.shuffle=shuffle

        if self.data_list is not None:
            #create queue with train/mask pairs
            self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
            self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
            self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
            self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                       shuffle=shuffle,
                                                       num_epochs=self.num_epochs)
            #changed to return filename as well
            self.image, self.label, self.image_file, self.label_file = read_images_from_disk(self.queue, self.input_size, random_scale)

        if self.inference_dir is not None:
            #create queue without labels
            L=os.listdir(self.inference_dir)
            self.inf_image_list=[os.path.join(self.inference_dir,l) for l in L]
            self.inf_images=tf.convert_to_tensor(self.inf_image_list, dtype=tf.string)
            self.inf_queue = tf.train.slice_input_producer([self.inf_images],
                                                       shuffle=shuffle,
                                                       num_epochs=self.num_epochs)
            self.inf_image,self.inf_image_file=read_unlabeled_images_from_disk(self.inf_queue,self.input_size,False)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        Args:
          num_elements: the batch size.
        Returns:
          Two tensors of size (batch_size, h, w, {3,1}) for images and masks.'''
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch

    def dequeue_inference(self,num_elements):
        '''Pack images into a batch.
        Args:
          num_elements: the batch size.
        Returns:
          1 tensor of size (batch_size, h, w, {3,1}) for image'''
        #image_batch = tf.train.batch([self.image], num_elements)

        image_batch, image_file_batch = tf.train.batch([self.inf_image,self.inf_image_file], num_elements)
        #x= tf.train.batch([self.inf_image], num_elements)
        #x= tf.train.batch([self.inf_image_file], num_elements)
        return image_batch, image_file_batch



