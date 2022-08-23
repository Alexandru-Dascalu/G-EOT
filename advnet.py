import random
import numpy as np
import tensorflow as tf
# gpu = tf.config.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_logical_device_configuration(
#     gpu,
#     [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])

import matplotlib.pyplot as plt

import Data
import Layers
import Nets
import target_model
import Preproc

wd = 1e-4
NoiseRange = 10.0


def create_generator(images, targets, num_experts, step, ifTest, layers):
    net = Layers.DepthwiseConv2D(Preproc.normalise_images(images), convChannels=3 * 16,
                                 convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='G_DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='G_SepConv96', dtype=tf.float32)
    layers.append(net)

    toadd = Layers.Conv2D(net.output, convChannels=192,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='G_SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='G_SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='G_SepConv192b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=384,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='G_SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)
    print(toadd.output.shape)

    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='G_SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='G_SepConv384b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=768,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='G_SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='G_SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='G_SepConv768b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    net = Layers.Activation(added, activation=Layers.ReLU, name='G_ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='G_SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.DeConv2D(net.output, convChannels=128,
                          convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv192', dtype=tf.float32)
    layers.append(net)
    subnets = []
    for idx in range(num_experts):
        subnet = Layers.DeConv2D(net.output, convChannels=64,
                                 convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv96_' + str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnet = Layers.DeConv2D(subnet.output, convChannels=32,
                                 convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv48_' + str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnet = Layers.Conv2D(subnet.output, convChannels=3,
                               convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                               convInit=Layers.XavierInit, convPadding='SAME',
                               biasInit=Layers.ConstInit(0.0),
                               batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               activation=Layers.ReLU,
                               reuse=tf.compat.v1.AUTO_REUSE, name='G_SepConv3_' + str(idx), dtype=tf.float32)
        layers.append(subnet)
        subnets.append(tf.expand_dims(subnet.output, axis=-1))
    subnets = tf.concat(subnets, axis=-1)
    weights = Layers.FullyConnected(tf.one_hot(targets, 10), outputSize=num_experts, weightInit=Layers.XavierInit,
                                    l2_constant=0.0,
                                    biasInit=Layers.ConstInit(0.0),
                                    activation=Layers.Softmax,
                                    reuse=tf.compat.v1.AUTO_REUSE, name='G_WeightsMoE', dtype=tf.float32)
    layers.append(weights)
    moe = tf.transpose(a=tf.transpose(a=subnets, perm=[1, 2, 3, 0, 4]) * weights.output, perm=[3, 0, 1, 2, 4])
    noises = (tf.nn.tanh(tf.reduce_sum(input_tensor=moe, axis=-1)) - 0.5) * NoiseRange * 2
    print('Shape of Noises: ', noises.shape)

    return noises


def create_simulator_SimpleNet(images, step, ifTest, layers):
    # define simulator with an architecture almost identical to SimpleNet in the paper
    net = Layers.DepthwiseConv2D(Preproc.normalise_images(tf.clip_by_value(images, 0, 255)), convChannels=3 * 16,
                                 convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv96', dtype=tf.float32)
    layers.append(net)

    toadd = Layers.Conv2D(net.output, convChannels=192,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='SepConv192b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=384,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv384b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=768,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv768b', dtype=tf.float32)
    layers.append(net)

    added = toadd.output + net.output

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv1024', dtype=tf.float32)
    layers.append(net)

    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, l2_constant=wd,
                                   biasInit=Layers.ConstInit(0.0),
                                   activation=Layers.Linear,
                                   reuse=tf.compat.v1.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    layers.append(logits)

    return logits.output


def create_simulatorG_SimpleNet(images, step, ifTest):
    # define simulator with an architecture almost identical to SimpleNet in the paper
    net = Layers.DepthwiseConv2D(Preproc.normalise_images(tf.clip_by_value(images, 0, 255)), convChannels=3 * 16,
                                 convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='DepthwiseConv3x16', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=96,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv96', dtype=tf.float32)

    toadd = Layers.Conv2D(net.output, convChannels=192,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv192Shortcut', dtype=tf.float32)

    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv192a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='SepConv192b', dtype=tf.float32)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=384,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv384Shortcut', dtype=tf.float32)

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv384a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv384b', dtype=tf.float32)

    added = toadd.output + net.output

    toadd = Layers.Conv2D(added, convChannels=768,
                          convKernel=[1, 1], convStride=[1, 1], l2_constant=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv768Shortcut', dtype=tf.float32)

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[2, 2], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv768a', dtype=tf.float32)
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv768b', dtype=tf.float32)

    added = toadd.output + net.output

    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU11024')
    net = Layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1], l2_constant=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv1024', dtype=tf.float32)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    logits = Layers.FullyConnected(net.output, outputSize=10, weightInit=Layers.XavierInit, l2_constant=wd,
                                   biasInit=Layers.ConstInit(0.0),
                                   activation=Layers.Linear,
                                   reuse=tf.compat.v1.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)

    return logits.output


hyper_params_imagenet = {'BatchSize': 8,
                         'NumSubnets': 10,
                         'NumPredictor': 1,
                         'NumGenerator': 1,
                         'NoiseDecay': 1e-5,
                         'LearningRate': 1e-3,
                         'MinLearningRate': 2 * 1e-5,
                         'DecayRate': 0.9,
                         'DecayAfter': 300,
                         'ValidateAfter': 300,
                         'TestSteps': 50,
                         'TotalSteps': 30000}


class AdvNet(Nets.Net):

    def __init__(self, image_shape, enemy, architecture, num_middle=2, hyper_params=None):
        Nets.Net.__init__(self)

        if hyper_params is None:
            hyper_params = hyper_params_imagenet

        self._init = False
        self._hyper_params = hyper_params

        # the targeted neural network model
        self._enemy = enemy
        with self._graph.as_default():

            # Inputs
            self._textures = tf.compat.v1.placeholder(dtype=tf.float32,
                                                      shape=[self._hyper_params['BatchSize']] + image_shape,
                                                      name='textures')
            self._labels = tf.compat.v1.placeholder(dtype=tf.int64, shape=[self._hyper_params['BatchSize']],
                                                    name='imagenet_labels')
            self._adversarial_targets = tf.compat.v1.placeholder(dtype=tf.int64,
                                                                 shape=[self._hyper_params['BatchSize']],
                                                                 name='target_labels')

            # define generator
            with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE) as scope:
                self._generator = create_generator(self._textures, self._adversarial_targets,
                                                   self._hyper_params['NumSubnets'], self._step,
                                                   self._ifTest, self._layers)
            self._noises = self._generator
            self._adversarial_textures = self._noises + self._textures

            # define simulator
            with tf.compat.v1.variable_scope('Predictor', reuse=tf.compat.v1.AUTO_REUSE) as scope:
                self._simulator = self.body(self._textures, architecture, num_middle, for_generator=False)
                # what is the point of this??? Why is the generator training against a different simulator, which is
                # not trained to match the target model? Why is one simulator trained on normal images, and another on
                # adversarial images?
                self._simulatorG = self.body(self._adversarial_textures, architecture, num_middle, for_generator=True)

            # define inference as hard label prediction of simulator on natural images
            self._inference = self.inference(self._simulator)
            # accuracy is how often simulator prediction matches the prediction of the target net
            self._accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self._inference, self._labels), tf.float32))

            self._loss = 0
            for elem in self._layers:
                if len(elem.losses) > 0:
                    for loss in elem.losses:
                        self._loss += loss

            # simulator loss matches simulator output against output of target model
            self._loss_simulator = self.loss(self._simulator, self._labels, name='lossP') + self._loss
            # generator trains to produce perturbations that make the simulator produce the desired target labels
            self._loss_generator = self.loss(self._simulatorG, self._adversarial_targets, name='lossG')
            self._loss_generator += self._hyper_params['NoiseDecay'] * tf.reduce_mean(
                input_tensor=tf.norm(tensor=self._noises)) + self._loss

            print(self.summary)
            # Saver
            self._saver = tf.compat.v1.train.Saver(max_to_keep=5)

    def body(self, images, architecture, num_middle=2, for_generator=False):
        # define body of simulator
        net_output = super().body(images, architecture, num_middle, for_generator)
        logits = Layers.FullyConnected(net_output, outputSize=1000, weightInit=Layers.XavierInit, l2_constant=wd,
                                       biasInit=Layers.ConstInit(0.0), activation=Layers.Linear,
                                       reuse=tf.compat.v1.AUTO_REUSE,
                                       name='P_FC_classes', dtype=tf.float32)
        if not for_generator:
            self._layers.append(logits)

        return logits.output

    def preproc(self, images):
        # normalise images fed into the simulator
        clipped = tf.clip_by_value(images, 0, 255)
        casted = tf.cast(clipped, tf.float32)
        standardized = tf.identity(casted / 127.5 - 1.0, name='training_standardized')

        return standardized

    def inference(self, logits):
        return tf.argmax(input=logits, axis=-1, name='inference')

    def loss(self, logits, labels, name='cross_entropy'):
        net = Layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output

    def train(self, data_generator, path_load=None, path_save=None):
        print("\n Begin Training: \n")
        with self._graph.as_default():
            self._lr = tf.compat.v1.train.exponential_decay(self._hyper_params['LearningRate'],
                                                            global_step=self._step,
                                                            decay_steps=self._hyper_params['DecayAfter'],
                                                            decay_rate=self._hyper_params['DecayRate'])
            self._lr += self._hyper_params['MinLearningRate']

            self._stepInc = tf.compat.v1.assign(self._step, self._step + 1)

            self._varsG = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
            self._varsS = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')

            # define optimisers
            self._optimizerS = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._loss_simulator,
                                                                                                 var_list=self._varsS)
            self._optimizerG = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-8)
            # clip generator optimiser gradients
            gradientsG = self._optimizerG.compute_gradients(self._loss_generator, var_list=self._varsG)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsG]
            self._optimizerG = self._optimizerG.apply_gradients(capped_gvs)

            # Initialize all
            self._sess.run(tf.compat.v1.global_variables_initializer())

            if path_load is not None:
                self.load(path_load)
            else:
                # warm up simulator to match predictions of target model on clean images
                print('Warming up. ')
                for idx in range(300):
                    data, _, _ = next(data_generator)
                    target_model_label = np.array(self._enemy.infer(data))
                    loss, accuracy, _ = self._sess.run([self._loss_simulator, self._accuracy, self._optimizerS],
                                                       feed_dict={self._textures: data,
                                                                  self._labels: target_model_label})
                    print('\rSimulator => Step: ', idx - 300,
                          '; Loss: %.3f' % loss,
                          '; Accuracy: %.3f' % accuracy,
                          end='')

                # evaluate warmed up simulator on test data
                warmupAccu = 0.0
                for idx in range(50):
                    data, _, _ = next(data_generator)
                    target_model_label = np.array(self._enemy.infer(data))
                    loss, accuracy, _ = \
                        self._sess.run([self._loss_simulator, self._accuracy, self._optimizerS],
                                       feed_dict={self._textures: data,
                                                  self._labels: target_model_label})
                    warmupAccu += accuracy
                warmupAccu = warmupAccu / 50
                print('\nWarmup Accuracy: ', warmupAccu)

            self.evaluate(data_generator)

            self._sess.run([self._phaseTrain])
            if path_save is not None:
                self.save(path_save)

            globalStep = 0
            # main training loop
            while globalStep < self._hyper_params['TotalSteps']:
                self._sess.run(self._stepInc)

                # train simulator for a couple of steps
                for _ in range(self._hyper_params['NumPredictor']):
                    # ground truth labels are not needed for training the simulator
                    data, _, target_label = next(data_generator)
                    # adds Random uniform noise to normal data
                    data = data + (np.random.rand(self._hyper_params['BatchSize'], 32, 32, 3) - 0.5) * 2 * NoiseRange

                    # perform one optimisation step to train simulator so it has the same predictions as the target
                    # model does on normal images with noise
                    target_model_labels = self._enemy.infer(data)
                    loss, accuracy, globalStep, _ = self._sess.run([self._loss_simulator, self._accuracy, self._step,
                                                                    self._optimizerS],
                                                                   feed_dict={self._textures: data,
                                                                              self._labels: target_model_labels})
                    print('\rSimulator => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; Accuracy: %.3f' % accuracy,
                          end='')

                    adversarial_images = self._sess.run(self._adversarial_textures,
                                                        feed_dict={self._textures: data,
                                                                   self._adversarial_targets: target_label})
                    # perform one optimisation step to train simulator so it has the same predictions as the target
                    # model does on adversarial images
                    target_model_labels = self._enemy.infer(adversarial_images)
                    loss, accuracy, globalStep, _ = self._sess.run([self._loss_simulator, self._accuracy, self._step,
                                                                    self._optimizerS],
                                                                   feed_dict={self._textures: adversarial_images,
                                                                              self._labels: target_model_labels})

                    self.simulator_loss_history.append(loss)
                    self.simulator_accuracy_history.append(accuracy)
                    print('\rSimulator => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; Accuracy: %.3f' % accuracy,
                          end='')

                # train generator for a couple of steps
                for _ in range(self._hyper_params['NumGenerator']):
                    data, _, target_label = next(data_generator)
                    target_model_labels = self._enemy.infer(data)

                    # make sure target label is different than what the target model already outputs for that image
                    for idx in range(data.shape[0]):
                        if target_model_labels[idx] == target_label[idx]:
                            tmp = random.randint(0, 9)
                            while tmp == target_model_labels[idx]:
                                tmp = random.randint(0, 9)
                            target_label[idx] = tmp

                    loss, adversarial_images, globalStep, _ = self._sess.run([self._loss_generator,
                                                                              self._adversarial_textures,
                                                                              self._step, self._optimizerG],
                                                                             feed_dict={self._textures: data,
                                                                                        self._adversarial_targets: target_label})
                    adversarial_predictions = self._enemy.infer(adversarial_images)
                    tfr = np.mean(target_label == adversarial_predictions)
                    ufr = np.mean(target_model_labels != adversarial_predictions)

                    self.generator_loss_history.append(loss)
                    self.generator_accuracy_history.append(tfr)
                    print('\rGenerator => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; TFR: %.3f' % tfr,
                          '; UFR: %.3f' % ufr,
                          end='')

                # evaluate on test every so ften
                if globalStep % self._hyper_params['ValidateAfter'] == 0:
                    self.evaluate(data_generator)
                    if path_save is not None:
                        self.save(path_save)
                        np.savez("./AttackCIFAR10/training_history", self.simulator_loss_history,
                                 self.simulator_accuracy_history, self.generator_loss_history,
                                 self.generator_accuracy_history, self.test_loss_history, self.test_accuracy_history)

                    self._sess.run([self._phaseTrain])

    def evaluate(self, test_data_generator, path=None):
        if path is not None:
            self.load(path)

        total_loss = 0.0
        total_tfr = 0.0
        total_ufr = 0.0

        self._sess.run([self._phaseTest])
        for _ in range(self._hyper_params['TestSteps']):
            data, _, target_labels = next(test_data_generator)
            target_model_labels = self._enemy.infer(data)

            # for each batch image, make sure target label is different than the predicted label by the target model
            for idx in range(data.shape[0]):
                if target_model_labels[idx] == target_labels[idx]:
                    tmp = random.randint(0, 9)
                    while tmp == target_model_labels[idx]:
                        tmp = random.randint(0, 9)
                    target_labels[idx] = tmp

            loss, adversarial_images = self._sess.run([self._loss_generator, self._adversarial_textures],
                                                      feed_dict={self._textures: data,
                                                                 self._adversarial_targets: target_labels})

            adversarial_images = adversarial_images.clip(0, 255).astype(np.uint8)
            adversarial_predictions = self._enemy.infer(adversarial_images)

            tfr = np.mean(target_labels == adversarial_predictions)
            ufr = np.mean(target_model_labels != adversarial_predictions)

            total_loss += loss
            total_tfr += tfr
            total_ufr += ufr

        total_loss /= self._hyper_params['TestSteps']
        total_tfr /= self._hyper_params['TestSteps']
        total_ufr /= self._hyper_params['TestSteps']

        self.test_loss_history.append(total_loss)
        self.test_accuracy_history.append(total_tfr)
        print('\nTest: Loss: ', total_loss,
              '; TFR: ', total_tfr,
              '; UFR: ', total_ufr)

    def sample(self, test_data_generator, path=None):
        if path is not None:
            self.load(path)

        self._sess.run([self._phaseTest])
        data, _, target = next(test_data_generator)

        target_model_labels = self._enemy.infer(data)
        for idx in range(data.shape[0]):
            if target_model_labels[idx] == target[idx]:
                tmp = random.randint(0, 9)
                while tmp == target_model_labels[idx]:
                    tmp = random.randint(0, 9)
                target[idx] = tmp

        loss, adversarial_images = self._sess.run([self._loss_generator, self._adversarial_textures],
                                                  feed_dict={self._textures: data,
                                                             self._adversarial_targets: target})
        adversarial_images = adversarial_images.clip(0, 255).astype(np.uint8)
        results = self._enemy.infer(adversarial_images)

        for idx in range(10):
            for jdx in range(3):
                # show sampled adversarial image
                plt.subplot(10, 6, idx * 6 + jdx * 2 + 1)
                plt.imshow(data[idx * 3 + jdx])
                plt.subplot(10, 6, idx * 6 + jdx * 2 + 2)
                plt.imshow(adversarial_images[idx * 3 + jdx])
                # print target model prediction on original image, the prediction on adversarial image, and target label
                print([target_model_labels[idx * 3 + jdx], results[idx * 3 + jdx], target[idx * 3 + jdx]])
        plt.show()

    def plot(self, genTest, path=None):
        if path is not None:
            self.load(path)

        data, label, target = next(genTest)

        tmpdata = []
        tmptarget = []

        for idx in range(10):
            while True:
                jdx = 0
                while jdx < data.shape[0]:
                    if label[jdx] == idx:
                        break
                    jdx += 1
                if jdx < data.shape[0]:
                    break
                else:
                    data, label, target = next(genTest)
            for ldx in range(10):
                if ldx != idx:
                    tmpdata.append(data[jdx][np.newaxis, :, :, :])
                    tmptarget.append(ldx)
        tmpdata = np.concatenate(tmpdata, axis=0)
        tmptarget = np.array(tmptarget)

        adversary = \
            self._sess.run(self._adversarial_textures,
                           feed_dict={self._textures: tmpdata,
                                      self._adversarial_targets: tmptarget})
        adversary = adversary.clip(0, 255).astype(np.uint8)

        kdx = 0
        for idx in range(10):
            jdx = 0
            while jdx < 10:
                if jdx == idx:
                    jdx += 1
                    continue
                plt.subplot(10, 10, idx * 10 + jdx + 1)
                plt.imshow(adversary[kdx, :, :, 0], cmap='gray')
                plt.axis('off')
                jdx += 1
                kdx += 1

        plt.show()


if __name__ == '__main__':
    enemy = target_model.NetImageNet([299, 299, 3], "SmallNet")
    tf.compat.v1.disable_eager_execution()
    enemy.load('./TargetModel/netcifar100.ckpt-32401')
    tf.compat.v1.enable_eager_execution()

    net = AdvNet([2048, 2048, 3], enemy, "SimpleNet")
    data_generator = Data.get_adversarial_data_generators(batch_size=hyper_params_imagenet['BatchSize'])

    net.train(data_generator, path_save='./AttackCIFAR10/netcifar10.ckpt')
    net.plot_training_history("Adversarial CIFAR10", net._hyper_params['ValidateAfter'])
