import tensorflow as tf


if __name__=='__main__':
    # Creates a graph.
    c = []
    for d in ['/gpu:0', '/gpu:1']:
        with tf.device(d):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)

    # Creates a session with log_device_placement set
    # to True.
    sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print sess.run(sum)
