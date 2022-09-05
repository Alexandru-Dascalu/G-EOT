import tensorflow as tf
from layers import conv2d_bn, depthwise_conv2d_bn, sep_conv2d_bn, deconv2d_bn

relu = tf.keras.activations.relu
NoiseRange = 10.0

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

def create_generator(num_experts):
    textures = tf.keras.layers.Input(shape=(2048, 2048, 3), dtype=tf.float32)
    targets = tf.keras.layers.Input(shape=(1,), dtype=tf.int64)
    x = simpleNet_encoder(textures)

    # add decoder part
    x = deconv2d_bn(x, filters=128, kernel_size=3, strides=2, activation=relu)
    subnets = []
    for idx in range(num_experts):
        subnet = deconv2d_bn(x, filters=64, kernel_size=3, strides=2, activation=relu)
        subnet = deconv2d_bn(subnet, filters=32, kernel_size=3, strides=2, activation=relu)
        subnet = deconv2d_bn(subnet, filters=16, kernel_size=3, strides=2, activation=relu)
        subnet = deconv2d_bn(subnet, filters=8, kernel_size=3, strides=2, activation=relu)
        subnet = deconv2d_bn(subnet, filters=3, kernel_size=3, strides=2, activation=relu)
        subnets.append(tf.expand_dims(subnet, axis=-1))

    subnets = tf.concat(subnets, axis=-1)
    weights = tf.keras.layers.Dense(units=num_experts, use_bias=True,
                                    kernel_regularizer=None,
                                    activation=tf.keras.activations.softmax)(tf.one_hot(targets, 1000))

    moe = tf.transpose(a=tf.transpose(a=subnets, perm=[1, 2, 3, 0, 4]) * weights, perm=[3, 0, 1, 2, 4])
    # why subtract 0.5 here? I think implementation is wrong here
    noises = (tf.nn.tanh(tf.reduce_sum(input_tensor=moe, axis=-1)) - 0.5) * NoiseRange * 2
    print('Shape of Noises: ', noises.shape)

    return tf.keras.Model(inputs=[textures, targets], outputs=noises)
