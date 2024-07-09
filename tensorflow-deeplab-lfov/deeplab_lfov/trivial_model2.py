import tensorflow as tf
from six.moves import cPickle
import numpy as np

from tf_utils import _linear

# Loading net skeleton with parameters name and shapes.
#with open("./util/net_skeleton.ckpt", "rb") as f:
#    net_skeleton = cPickle.load(f)


'''
We just tried logistic regression on average value as a low res model.
Now try some fairly simple classification CNN

'''


n_classes = 2#21
# All convolutional and pooling operations are applied using kernels of size 3x3; 
# padding is added so that the output of the same size as the input.
ks = 3

def create_variable(name, shape):
    """Create a convolution filter variable of the given name and shape,
       and initialise it using Xavier initialisation
       (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    """Create a bias variable of the given name and shape,
       and initialise it to zero.
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable



class VGG(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.

    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """

    def __init__(self,image_batch, label_batch=None):
        """Create the model.

        Args:
          weights_path: the path to the cpkt file with dictionary of weights from .caffemodel.
        """

        print 'using trivial 2'

        if not label_batch is None:
            self.calc_loss( image_batch, label_batch)

        else:
            self.calc_preds( image_batch )


    def _create_network(self, input_data, is_training):
        """Construct DeepLab-LargeFOV network.

        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.

        Returns:
          A downsampled segmentation mask.
        """

        self.input_data=input_data
        relu=tf.nn.relu

        #bs,300,300,3
        network = input_data

        #bs,150,150,3
        network = tf.layers.average_pooling2d(network, pool_size=2,
                                              strides=2)
        self.n0=network

        #bs,74,74,32
        network=tf.layers.conv2d(network, filters=32, kernel_size=3, strides=2,activation=relu)

        #bs,36,36,64
        network=tf.layers.conv2d(network, filters=64, kernel_size=3, strides=2,activation=relu)

        #bs,12,12,64
        network=tf.layers.max_pooling2d(network,pool_size=3,strides=3)

        #bs,5,5,128
        network=tf.layers.conv2d(network, filters=128, kernel_size=3, strides=2,activation=relu)

        #bs,2,2,128
        network=tf.layers.max_pooling2d(network,pool_size=2,strides=2)

        #bs,512
        network=tf.reshape(network, [-1,2*2*128])#hack for now

        network=relu( _linear( network, 512 ) )

        #network=tf.layers.dropout

        self.n1=network
        logits = _linear(network, n_classes, scope='proj')
        return logits


    def prepare_label(self, input_batch):
        with tf.name_scope('label_encode'):

            ##old code
            #input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # As labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.
            self.ib1=input_batch#-1,300,300

            ##new hack
            #label as 1 if any lesion present
            input_batch=tf.reduce_max(input_batch,axis=[1,2])
            self.ib2=input_batch #(-1,)

            input_batch = tf.one_hot(input_batch, depth=n_classes)
            self.ib3=input_batch#(-1,2)
        return input_batch

    def calc_preds(self, input_batch):
        """Create the network and run inference on the input batch.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        input_batch=tf.cast(input_batch, tf.float32)


        #with tf.variable_scope('vgg',reuse=True):#has to already be created
        with tf.variable_scope('vgg'):
            self.logits = self._create_network(input_batch, is_training=False)
        #self.logits = tf.image.resize_bilinear(self.ds_logits, tf.shape(input_batch)[1:3,])
        self.probs = tf.nn.softmax(self.logits)
        y_hat = tf.argmax(self.logits, dimension=-1)
        self.preds = tf.cast(y_hat, tf.uint8)


    def calc_loss(self, img_batch, label_batch):
        """Create the network, run inference on the input batch and compute loss.
        #Run this before preds#

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Pixel-wise softmax loss.
        """
        #TODO modulate dropout prob
        img_batch=tf.cast(img_batch, tf.float32)
        with tf.variable_scope('vgg',reuse=False):
            raw_output = self._create_network(img_batch, is_training=True)
            self.logits=raw_output

        self.probs = tf.nn.softmax(self.logits)
        #y_hat = tf.argmax(self.logits, dimension=-1)
        #self.preds = tf.cast(y_hat, tf.uint8)


        prediction = tf.reshape(raw_output, [-1, n_classes])

        self.prediction=prediction
        print 'predictionshape:',prediction.get_shape()
        #self.probs=tf.nn.softmax(prediction)#no

        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch)
        gt = tf.reshape(label_batch, [-1, n_classes])

        self.gt=gt

        #very experimental
        #warning overflow e^(-x), x<0

        #Don't be more than 0.6 confident in y=1
        adjust_probs=self.probs*0.6

        batch_loss= -gt*tf.log(adjust_probs) - (1-gt)*tf.log(1-adjust_probs)

        #batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        self.loss = tf.reduce_mean(batch_loss)

