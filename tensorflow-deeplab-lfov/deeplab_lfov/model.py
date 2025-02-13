import tensorflow as tf
from six.moves import cPickle

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
    ['fc6/w', (3, 3, 512, 1024)],
    ['fc6/b', (1024,)],
    ['fc7/w', (1, 1, 1024, 1024)],
    ['fc7/b', (1024,)],
    ['fc8_voc12/w', (1, 1, 1024, 2)],#21<-2
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
                 [2, 2, 2],#atrous
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




class DeepLabLFOVModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.

    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """

    def __init__(self,image_batch=None, label_batch=None, weights_path=None):
#    def __init__(self, weights_path=None):
        """Create the model.

        Args:
          weights_path: the path to the cpkt file with dictionary of weights from .caffemodel.
        """
        self.variables = self._create_variables(weights_path)

        if image_batch is not None:
            if not label_batch is None:
                #appologies to coding
                self.loss=self.loss(image_batch,label_batch)
                #self.calc_loss( image_batch, label_batch)
            else:
                self.preds=self.preds(image_batch)
                #self.calc_preds( image_batch )

    def _create_variables(self, weights_path):
        """Create all variables used by the network.
        This allows to share them between multiple calls
        to the loss function.

        Args:
          weights_path: the path to the ckpt file with dictionary of weights from .caffemodel.
                        If none, initialise all variables randomly.

        Returns:
          A dictionary with all variables.
        """
        var = list()
        index = 0

        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f) # Load pre-trained weights.
                for name, shape in net_skeleton:
                    var.append(tf.Variable(weights[name],
                                           name=name))
                del weights
        else:
            # Initialise all weights randomly with the Xavier scheme,
            # and 
            # all biases to 0's.
            for name, shape in net_skeleton:
                if "/w" in name: # Weight filter.
                    w = create_variable(name, list(shape))
                    var.append(w)
                else:
                    b = create_bias_variable(name, list(shape))
                    var.append(b)
        return var


    def _create_network(self, input_batch, keep_prob):
        """Construct DeepLab-LargeFOV network.

        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.

        Returns:
          A downsampled segmentation mask.
        """
        current = input_batch

        v_idx = 0 # Index variable.

        # Last block is the classification layer.
        for b_idx in xrange(len(dilations) - 1):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = self.variables[v_idx * 2]
                b = self.variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
            # Optional pooling and dropout after each block.
            if b_idx < 3:
                current = tf.nn.max_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current,
                             ksize=[1, ks, ks, 1],
                             strides=[1, 1, 1, 1],
                             padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                current = tf.nn.avg_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)

        # Classification layer; no ReLU.
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        return current

    def prepare_label(self, input_batch, new_size):
        """Resize masks and perform one-hot encoding.

        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.

        Returns:
          Outputs a tensor of shape [batch_size h w n_classes]
          with last dimension comprised of 0's and 1's only.
        """
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # As labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.

            ####NEW##
            #input_batch = tf.one_hot(input_batch, depth=21)
            input_batch = tf.one_hot(input_batch, depth=2)
        return input_batch

    def preds(self, input_batch):
        """Create the network and run inference on the input batch.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        self.ds_logits = self._create_network(tf.cast(input_batch, tf.float32), keep_prob=tf.constant(1.0))
        self.logits = tf.image.resize_bilinear(self.ds_logits, tf.shape(input_batch)[1:3,])
        self.probs=tf.nn.softmax(self.logits)
        y_hat = tf.argmax(self.logits, dimension=3)
        y_hat = tf.expand_dims(y_hat, dim=3) # Create 4D-tensor.
        return tf.cast(y_hat, tf.uint8)


    def loss(self, img_batch, label_batch):
        """Create the network, run inference on the input batch and compute loss.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Pixel-wise softmax loss.
        """
        #TODO modulate dropout prob
        raw_output = self._create_network(tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))
        prediction = tf.reshape(raw_output, [-1, n_classes])

        #self.probs=tf.nn.softmax(prediction)#no

        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, n_classes])

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        return reduced_loss
