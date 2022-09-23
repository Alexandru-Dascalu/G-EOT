import tensorflow as tf
import config

initialiser = tf.keras.initializers.Orthogonal()
l2_regulariser = tf.keras.regularizers.L2(config.hyper_params['LayerRegularisationWeight'])


def conv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True,
                               activation=activation, kernel_regularizer=l2_regulariser,
                               kernel_initializer=initialiser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def depthwise_conv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    channels_in = x.shape[3]
    assert filters % channels_in == 0
    depth_multiplier = int(filters / channels_in)

    x = tf.keras.layers.DepthwiseConv2D(depth_multiplier=depth_multiplier, kernel_size=kernel_size, strides=strides,
                                        padding='same', use_bias=True, activation=activation,
                                        depthwise_regularizer=l2_regulariser,
                                        depthwise_initializer=initialiser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def sep_conv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    x = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                        use_bias=True, activation=activation, depthwise_regularizer=l2_regulariser,
                                        pointwise_regularizer=l2_regulariser,
                                        depthwise_initializer=initialiser, pointwise_initializer=initialiser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def deconv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                        use_bias=True, activation=activation, kernel_regularizer=l2_regulariser,
                                        kernel_initializer=initialiser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x
