import layers


def get_Simple_Net_encoder(normalised_textures, step, ifTest, layers_list, name_prefix=""):
    # textures are 2048x2048
    net = layers.Conv2D(normalised_textures, convChannels=12, convKernel=[3, 3], convStride=[2, 2],
                        batch_norm=True, step=step, ifTest=ifTest,
                        activation=layers.ReLU, name='{}Conv32'.format(name_prefix))
    layers_list.append(net)

    # textures are 1024x1024
    net = layers.Conv2D(net.output, convChannels=24, convKernel=[3, 3],
                        batch_norm=True, step=step, ifTest=ifTest,
                        activation=layers.ReLU,
                        name='{}Conv48'.format(name_prefix))
    layers_list.append(net)
    net = layers.DepthwiseConv2D(net.output, convChannels=48, convKernel=[3, 3], convStride=[2, 2],
                                 batch_norm=True, step=step, ifTest=ifTest,
                                 activation=layers.ReLU,
                                 name='{}DepthwiseConv48'.format(name_prefix))
    layers_list.append(net)

    # textures are 512x512
    net = layers.SepConv2D(net.output, convChannels=96,
                           convKernel=[3, 3],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv96'.format(name_prefix), )
    layers_list.append(net)

    toadd = layers.Conv2D(net.output, convChannels=192,
                          convKernel=[1, 1],
                          batch_norm=True, step=step, ifTest=ifTest,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='{}SepConv192Shortcut'.format(name_prefix))
    layers_list.append(toadd)

    net = layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[2, 2],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv192a'.format(name_prefix))
    layers_list.append(net)
    # textures are 256x256
    # why does this not have activation?
    net = layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3],
                           batch_norm=True, step=step, ifTest=ifTest,
                           name='{}SepConv192b'.format(name_prefix))
    layers_list.append(net)
    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=384, convKernel=[1, 1],
                          batch_norm=True, step=step, ifTest=ifTest,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='{}SepConv384Shortcut'.format(name_prefix))
    layers_list.append(toadd)

    # why activate this again?
    net = layers.Activation(added, activation=layers.ReLU, name='{}ReLU384'.format(name_prefix))
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[2, 2],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv384a'.format(name_prefix))
    # textures are 128x128
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=384, convKernel=[3, 3],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv384b'.format(name_prefix))
    layers_list.append(net)
    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=768, convKernel=[1, 1],
                          batch_norm=True, step=step, ifTest=ifTest,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='{}SepConv768Shortcut'.format(name_prefix))
    layers_list.append(toadd)

    # why activate this again?
    net = layers.Activation(added, activation=layers.ReLU, name='{}ReLU768'.format(name_prefix))
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[2, 2],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv768a'.format(name_prefix))
    layers_list.append(net)
    # textures are 64x64 now
    net = layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[1, 1],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv768b'.format(name_prefix))
    layers_list.append(net)
    added = toadd.output + net.output

    # why activate this? both toadd and net had RELU activation
    net = layers.Activation(added, activation=layers.ReLU, name='{}ReLU11024'.format(name_prefix))
    layers_list.append(net)

    net = layers.SepConv2D(net.output, convChannels=1024, convKernel=[3, 3],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv1024a'.format(name_prefix))
    layers_list.append(net)
    net = layers.Pooling(net.output, pool=layers.MaxPool,
                         size=[3, 3], stride=[2, 2],
                         name='{}MaxPoolEnd'.format(name_prefix))
    layers_list.append(net)
    # textures are 32x32
    net = layers.SepConv2D(net.output, convChannels=1024, convKernel=[3, 3],
                           batch_norm=True, step=step, ifTest=ifTest,
                           activation=layers.ReLU,
                           name='{}SepConv1024b'.format(name_prefix))
    layers_list.append(net)

    return net
