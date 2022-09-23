import tensorflow as tf

import layers
from layers import conv2d_bn, depthwise_conv2d_bn, sep_conv2d_bn, deconv2d_bn

relu = tf.keras.activations.relu

# input must be normalised textures
def simpleNet_encoder(textures):
    # textures are 2048x2048
    x = tf.keras.layers.AvgPool2D(pool_size=8)(textures)
    # textures are 256x256 now
    x = conv2d_bn(x, filters=32, kernel_size=3, strides=2, activation=relu)
    x = depthwise_conv2d_bn(x, filters=64, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=96, kernel_size=3, activation=relu)

    # tensor is 128x128x96 now
    # three blocks with skip connections
    toadd = conv2d_bn(x, filters=192, kernel_size=1, activation=None)
    toadd = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(toadd)

    x = sep_conv2d_bn(x, filters=192, kernel_size=3, strides=2, activation=relu)
    x = sep_conv2d_bn(x, filters=192, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 64x64x192 now
    # second skip connection block
    toadd = conv2d_bn(x, filters=384, kernel_size=1, activation=None)
    toadd = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(toadd)

    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=384, kernel_size=3, strides=2, activation=relu)
    x = sep_conv2d_bn(x, filters=384, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 32x32x384 now
    # third skip connection block
    toadd = conv2d_bn(x, filters=768, kernel_size=1, activation=None)
    toadd = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(toadd)

    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, strides=2, activation=relu)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=None)
    toadd = tf.keras.layers.Add()([x, toadd])

    # tensor is 16x16x768 now
    # entering middle flow, with two skip blocks, no spatial reduction, 768 kernels
    x = tf.keras.layers.Activation(activation=relu)(toadd)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=None)
    toadd = tf.keras.layers.Add()([x, toadd])

    x = tf.keras.layers.Activation(activation=relu)(toadd)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=768, kernel_size=3, activation=None)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 16x16x768 now
    # entering exit flow, with one skip block and on final separable convolution
    toadd = conv2d_bn(x, filters=1536, kernel_size=1, strides=2, activation=None)

    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=1536, kernel_size=3, activation=relu)
    x = sep_conv2d_bn(x, filters=1536, kernel_size=3, activation=None)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.Add()([x, toadd])

    # tensor is 8x8x1024 now
    # activate after adding skip connection toadd with unactivated x tensor
    x = tf.keras.layers.Activation(activation=relu)(x)
    x = sep_conv2d_bn(x, filters=2048, kernel_size=3, activation=relu)

    return x


def create_generator(num_experts):
    textures = tf.keras.layers.Input(shape=(2048, 2048, 3), dtype=tf.float32)
    targets = tf.keras.layers.Input(shape=(), dtype=tf.int64)
    x = simpleNet_encoder(textures)

    # add decoder part
    x = deconv2d_bn(x, filters=256, kernel_size=3, strides=2, activation=relu)
    subnets = []
    for idx in range(num_experts):
        subnet = deconv2d_bn(x, filters=128, kernel_size=3, strides=2, activation=relu)
        subnet = deconv2d_bn(subnet, filters=64, kernel_size=3, strides=2, activation=relu)
        subnet = deconv2d_bn(subnet, filters=32, kernel_size=3, strides=2, activation=relu)
        subnet = deconv2d_bn(subnet, filters=3, kernel_size=3, strides=2, activation=relu)
        subnets.append(tf.expand_dims(subnet, axis=-1))

    subnets = tf.concat(subnets, axis=-1)
    weights = tf.keras.layers.Dense(units=num_experts, use_bias=True,
                                    kernel_regularizer=None, kernel_initializer=layers.initialiser,
                                    activation=tf.keras.activations.softmax)(tf.one_hot(targets, 1000))

    subnets = tf.transpose(a=subnets, perm=[1, 2, 3, 0, 4])
    moe = tf.transpose(a= subnets * weights, perm=[3, 0, 1, 2, 4])

    noises = (tf.nn.tanh(tf.reduce_sum(input_tensor=moe, axis=-1)) - 0.5) * 2 * 25
    noises = tf.keras.layers.UpSampling2D(size=8, interpolation="bicubic")(noises)
    noises = noises / 255
    print('Shape of Noises: ', noises.shape)

    return tf.keras.Model(inputs=[textures, targets], outputs=noises, name="generator")
