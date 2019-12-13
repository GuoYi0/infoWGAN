import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops


def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret


def identity(x):
    return x


def batch_norm(x, is_training, channels=None, name="bn", data_format="NHWC", trainable=True, bias_on_bn=True,
               scale_on_bn=True, decay=0.95, gradient_on_mean_varience=True):
    assert data_format in ("NCHW", "NHWC")
    axis = [0, 2, 3] if data_format == "NCHW" else [0, 1, 2]
    shape = [channels]
    with tf.variable_scope(name):
        moving_mean = tf.get_variable(name='moving_mean', shape=shape, trainable=False,
                                      initializer=tf.zeros_initializer)
        moving_variance = tf.get_variable(name='moving_variance', shape=shape, trainable=False,
                                          initializer=tf.ones_initializer)
        mean_, variance_ = tf.nn.moments(x, axes=axis, keep_dims=False, name='moments')

        update_moving_mean = moving_averages.assign_moving_average(variable=moving_mean, value=mean_,
                                                                   decay=decay, zero_debias=False,
                                                                   name='moving_mean_op')
        update_moving_variance = moving_averages.assign_moving_average(variable=moving_variance, value=variance_,
                                                                       decay=decay, zero_debias=False,
                                                                       name='moving_variance_op')

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
        if not gradient_on_mean_varience:
            mean_ = tf.stop_gradient(mean_)
            variance_ = tf.stop_gradient(variance_)
        mean, variance = control_flow_ops.cond(
            is_training, lambda: (mean_, variance_),
            lambda: (moving_mean, moving_variance))

        beta, gamma = None, None
        if bias_on_bn:
            beta = tf.get_variable(name='beta', shape=[channels], trainable=trainable, initializer=tf.zeros_initializer)
        if scale_on_bn:
            gamma = tf.get_variable(name='gamma', shape=[channels], trainable=trainable,
                                    initializer=tf.ones_initializer)

        if data_format == "NCHW":
            mean = tf.reshape(mean, (1, channels, 1, 1))
            variance = tf.reshape(variance, (1, channels, 1, 1))
            beta = tf.reshape(beta, (1, channels, 1, 1))
        else:
            mean = tf.reshape(mean, (1, 1, 1, channels))
            variance = tf.reshape(variance, (1, 1, 1, channels))
            gamma = tf.reshape(gamma, (1, 1, 1, channels))

        x = tf.nn.batch_normalization(x, mean=mean, variance=variance,
                                      scale=gamma, offset=beta, variance_epsilon=1e-9)
        return x
