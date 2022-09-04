import tensorflow as tf
from layers import conv2d_bn, depthwise_conv2d_bn, sep_conv2d_bn

relu = tf.keras.activations.relu


# input must be normalised textures
def simpleNet_encoder(textures):
    # textures are 2048x2048
    x = conv2d_bn(textures, filters=12, kernel_size=3, strides=2, activation=relu)
    # textures are 1024x1024
    x = conv2d_bn(x, filters=24, kernel_size=3, activation=relu)
    x = depthwise_conv2d_bn(x, filters=48, kernel_size=3, strides=2, activation=relu)
    # textures are 512x512 now
    x = sep_conv2d_bn(x, filters=96, kernel_size=3, activation=relu)

    # first skip block
    toadd = conv2d_bn(x, filters=192, kernel_size=1, activation=None)
    toadd = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(toadd)

    x = sep_conv2d_bn(x, filters=192, kernel_size=3, strides=2, activation=relu)
    # textures are 256x256
    x = sep_conv2d_bn(x, filters=192, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # second skip block
    toadd = conv2d_bn(x, filters=384, kernel_size=1, activation=None)
    toadd = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(toadd)

    x = tf.keras.layers.Activation(relu)(x)
    x = sep_conv2d_bn(x, filters=384, kernel_size=3, strides=2, activation=relu)
    # textures are 128x128
    x = sep_conv2d_bn(x, filters=384, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # third skip block
    toadd = conv2d_bn(x, filters=768, kernel_size=1, activation=None)
    toadd = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(toadd)

    x = tf.keras.layers.Activation(relu)(x)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, strides=2, activation=relu)
    # textures are 64x64 now
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    x = tf.keras.layers.Activation(relu)(x)
    x = sep_conv2d_bn(x, filters=1024, kernel_size=3, activation=relu)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # textures are 32x32 now
    x = sep_conv2d_bn(x, filters=1024, kernel_size=3, activation=relu)

    return x
