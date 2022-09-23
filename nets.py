import tensorflow as tf
from layers import conv2d_bn, sep_conv2d_bn, depthwise_conv2d_bn
import config

relu = tf.keras.activations.relu


def create_simulator(architecture):
    """
    Defines the body of the NN and adds all layers to the layers list of the NN.

    Parameters
    ----------
    architecture : str
        String naming the architecture to use for the simulator of this model.

    Returns
    ----------
    outputs
        A tf.keras.Model representing a model with the required architecture
    """

    if architecture == "SimpleNet":
        return get_Simple_Net()
    else:
        raise ValueError("Invalid simulator architecture argument!")


def get_Simple_Net():
    # input images must have values between -1 and 1
    input_images = tf.keras.layers.Input(shape=(299, 299, 3), dtype=tf.float32)
    # initial three layers in entry flow
    x = conv2d_bn(input_images, filters=24, kernel_size=3, strides=2, activation=relu)
    x = depthwise_conv2d_bn(x, filters=48, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=96, kernel_size=3, activation=relu)

    # tensor is 150x150x96 now
    # three blocks with skip connections
    toadd = conv2d_bn(x, filters=192, kernel_size=1, strides=2, activation=None)

    x = sep_conv2d_bn(x, filters=192, kernel_size=3, strides=2, activation=relu)
    x = sep_conv2d_bn(x, filters=192, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 75x75x192 now
    # second skip connection block
    toadd = conv2d_bn(x, filters=384, kernel_size=1, strides=2, activation=None)

    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=384, kernel_size=3, strides=2, activation=relu)
    x = sep_conv2d_bn(x, filters=384, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 38x38x384 now
    # third skip connection block
    toadd = conv2d_bn(x, filters=768, kernel_size=1, strides=2, activation=None)

    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, strides=2, activation=relu)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=None)
    toadd = tf.keras.layers.Add()([x, toadd])

    # tensor is 19x19x768 now
    # entering middle flow, with two skip blocks, no spatial reduction, 768 kernels
    x = tf.keras.layers.Activation(activation=relu)(toadd)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=None)
    toadd = tf.keras.layers.Add()([x, toadd])

    x = tf.keras.layers.Activation(activation=relu)(toadd)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 19x19x768 now
    # entering exit flow, with one skip block and on final separable convolution
    toadd = conv2d_bn(x, filters=1024, kernel_size=1, strides=2, activation=None)

    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=1024, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=1024, kernel_size=3, activation=None)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 10x10x1024 now
    # activate after adding skip connection toadd with unactivated x tensor
    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=1536, kernel_size=3, activation=relu)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    # logits layer
    l2_regularisation_constant = config.hyper_params['LayerRegularisationWeight']
    x = tf.keras.layers.Dense(units=1000, kernel_regularizer=tf.keras.regularizers.L2(l2_regularisation_constant),
                              activation=None)(x)

    return tf.keras.Model(inputs=input_images, outputs=x, name="SimpleNet simulator")
