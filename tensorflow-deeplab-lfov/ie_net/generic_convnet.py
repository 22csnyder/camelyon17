import tensorflow as tf
from tf_utils import _linear


conv_init= tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
bias_init=tf.constant_initializer(value=0.0, dtype=tf.float32)
def get_conv_variables(name, shape):
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    with tf.variable_scope(name or "conv"):
        W = tf.get_variable("Matrix",shape,initializer= conv_init)
        b=tf.get_variable('Bias', [1,shape[-1]],initializer=bias_init)
    return W,b
    #with vs.variable_scope(name or "conv"):
    #    W = vs.get_variable("Matrix",shape,initializer= conv_init)
    #    b=vs.get_variable('Bias', [1,shape[-1]],initializer=bias_init)

def conv_layer(inputs, ker_shape, strides,name):
    W,b=get_conv_variables(name, ker_shape)
    act=tf.nn.conv2d(inputs,W,strides=strides,padding='VALID',name=name)
    return tf.nn.relu( act + b )


def print_shape(act,debug):
    if debug:
        print act.get_shape().as_list()

class GenericConvNet(object):

    def __init__(self, is_training, n_classes=2):
        self.n_classes=n_classes
        self.is_training=is_training
    def __call__(self,act):
        debug=True
        #256x256x3 inputs after preprocessing

        act = conv_layer(act, [3,3,3,16],strides=[1,2,2,1], name='conv1_1')
        print_shape(act,debug)
        act = tf.nn.max_pool( act, ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='mp1')
        print_shape(act,debug)

        act = conv_layer(act, [3,3,16,32],strides=[1,2,2,1], name='conv2_1')
        print_shape(act,debug)
        act = conv_layer(act, [3,3,32,64],strides=[1,2,2,1], name='conv2_2')
        print_shape(act,debug)
        act = tf.nn.max_pool( act, ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='mp2')
        print_shape(act,debug)

        act = conv_layer(act, [3,3,64,128],strides=[1,1,1,1], name='conv3_1')
        print_shape(act,debug)
        act = tf.nn.max_pool( act,ksize=[1,3,3,1],strides=[1,1,1,1],padding='VALID',name='mp3')
        print_shape(act,debug)

        act = tf.contrib.layers.flatten(act,scope='flatten4')
        print_shape(act,debug)
        act = tf.nn.relu( _linear( act, 512, scope='fc4' ) )
        print_shape(act,debug)
        if self.is_training:
            act = tf.nn.dropout(act, 0.5)
        print_shape(act,debug)

        act = _linear( act, self.n_classes, scope='proj')
        print_shape(act,debug)

        return act


