import tensorflow as tf
from six.moves import cPickle
import numpy as np

# Loading net skeleton with parameters name and shapes.
#with open("./util/net_skeleton.ckpt", "rb") as f:
#    net_skeleton = cPickle.load(f)

##weights## (these were in net_skeleton)
#I moved them here to change the last one
net_skeleton= [['conv1_1/w', (3, 3, 3, 64)],
    ['conv1_1/b', (64,)],
    ['conv1_2/w', (3, 3, 64, 64)],
    ['conv1_2/b', (64,)],
    ['conv2_1/w', (3, 3, 64, 128)],
    ['conv2_1/b', (128,)],
    ['conv2_2/w', (3, 3, 128, 128)],
    ['conv2_2/b', (128,)],
    ['conv3_1/w', (3, 3, 128, 256)],
    ['conv3_1/b', (256,)],
    ['conv3_2/w', (3, 3, 256, 256)],
    ['conv3_2/b', (256,)],
    ['conv3_3/w', (3, 3, 256, 256)],
    ['conv3_3/b', (256,)],
    ['conv4_1/w', (3, 3, 256, 512)],
    ['conv4_1/b', (512,)],
    ['conv4_2/w', (3, 3, 512, 512)],
    ['conv4_2/b', (512,)],
    ['conv4_3/w', (3, 3, 512, 512)],
    ['conv4_3/b', (512,)],

    ['conv5_1/w', (3, 3, 512, 512)],
    ['conv5_1/b', (512,)],
    ['conv5_2/w', (3, 3, 512, 512)],
    ['conv5_2/b', (512,)],
    ['conv5_3/w', (3, 3, 512, 512)],
    ['conv5_3/b', (512,)],

    ['fc6/w', (3, 3, 512, 4096)],#1024<-4096
    ['fc6/b', (4096,)],
    ['fc7/w', (1, 1, 4096, 4096)],#1024<-4096
    ['fc7/b', (4096,)],
    ['fc8_voc12/w', (1, 1, 4096, 2)],#21<-2#1024<-4096
    ['fc8_voc12/b', (2,)]]#21<-2


# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)

##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=2) -> [pixel-wise softmax loss].#21<-2 in edit
num_layers    = [2, 2, 3, 3, 3, 1, 1, 1]
dilations     = [[1, 1],
                 [1, 1],
                 [1, 1, 1],
                 [1, 1, 1],

                 [1,1,1],#[2, 2, 2],#atrous
                 [12], #atrous
                 [1],
                 [1]]
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
        network = input_data
        network = tf.layers.conv2d(network, 64, 3, 2,'same', activation=relu)
        network = tf.layers.conv2d(network, 64, 3, 2, activation=relu)
        network = tf.layers.max_pooling2d(network, 2, 2)

        network = tf.layers.conv2d(network, 128, 3, 2, 'same', activation=relu)
        network = tf.layers.conv2d(network, 128, 3, 2, 'same', activation=relu)
        network = tf.layers.max_pooling2d(network, 2, 2)

        network = tf.layers.conv2d(network, 256, 3, 1, 'same', activation=relu)
        network = tf.layers.conv2d(network, 256, 3, 1, 'same', activation=relu)
        network = tf.layers.conv2d(network, 256, 3, 1, 'same', activation=relu)
        network = tf.layers.max_pooling2d(network, 2, 2)

        network = tf.layers.conv2d(network, 512, 3, 1, 'same', activation=relu)
        network = tf.layers.conv2d(network, 512, 3, 1, 'same', activation=relu)
        network = tf.layers.conv2d(network, 512, 3, 1, 'same', activation=relu)
        network = tf.layers.max_pooling2d(network, 2, 2)

        shape=network.get_shape().as_list()
        n_inputs=int(np.prod(shape[1:]))
        network=tf.reshape(network,[-1,n_inputs])

        network = tf.layers.dense(network, 4096, activation=relu)
        network = tf.layers.dropout(network, 0.5, training=is_training)

        network = tf.layers.dense(network, 4096, activation=relu)
        network = tf.layers.dropout(network, 0.5, training=is_training)

        logits = tf.layers.dense(network, n_classes, activation=None)

        return logits

    def prepare_label(self, input_batch):
        """Resize masks and perform one-hot encoding.

        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.

        Returns:
          Outputs a tensor of shape [batch_size h w n_classes]
          with last dimension comprised of 0's and 1's only.
        """
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


        with tf.variable_scope('vgg',reuse=True):
            self.logits = self._create_network(input_batch, is_training=False)
        #self.logits = tf.image.resize_bilinear(self.ds_logits, tf.shape(input_batch)[1:3,])
        self.probs = tf.nn.softmax(self.logits)
        y_hat = tf.argmax(self.logits, dimension=-1)
        self.pred = tf.cast(y_hat, tf.uint8)


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
        prediction = tf.reshape(raw_output, [-1, n_classes])

        #self.probs=tf.nn.softmax(prediction)#no

        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch)
        gt = tf.reshape(label_batch, [-1, n_classes])

        self.gt=gt

        # Pixel-wise softmax loss.
        batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        self.loss = tf.reduce_mean(batch_loss)

