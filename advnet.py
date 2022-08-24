
import tensorflow as tf
# gpu = tf.config.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_logical_device_configuration(
#     gpu,
#     [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])

import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import random
import numpy as np

import data
import layers
import nets
import encoders
import target_model
import preproc
from uv_renderer import UVRenderer
from config import cfg

NoiseRange = 10.0


def create_encoder(textures, step, ifTest, layers_list):
    return encoders.get_Simple_Net_encoder(textures, step, ifTest, layers_list, name_prefix="G_")


def create_generator(textures, targets, num_experts, step, ifTest, layers_list):
    textures = preproc.normalise_images(textures)
    encoder = create_encoder(textures, step, ifTest, layers_list)

    net = layers.DeConv2D(encoder.output, convChannels=128,
                          convKernel=[3, 3], convStride=[2, 2],
                          batch_norm=True, step=step, ifTest=ifTest,
                          activation=layers.ReLU,
                          reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv128')
    layers_list.append(net)

    subnets = []
    for idx in range(num_experts):
        subnet = layers.DeConv2D(net.output, convChannels=64,
                                 convKernel=[3, 3], convStride=[2, 2],
                                 batch_norm=True, step=step, ifTest=ifTest,
                                 activation=layers.ReLU,
                                 reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv64_' + str(idx))
        layers_list.append(subnet)
        subnet = layers.DeConv2D(subnet.output, convChannels=32,
                                 convKernel=[3, 3], convStride=[2, 2],
                                 batch_norm=True, step=step, ifTest=ifTest,
                                 activation=layers.ReLU,
                                 reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv32_' + str(idx))
        layers_list.append(subnet)
        subnet = layers.DeConv2D(subnet.output, convChannels=16,
                                 convKernel=[3, 3], convStride=[2, 2],
                                 batch_norm=True, step=step, ifTest=ifTest,
                                 activation=layers.ReLU,
                                 reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv16_' + str(idx))
        layers_list.append(subnet)
        subnet = layers.DeConv2D(subnet.output, convChannels=8,
                                 convKernel=[3, 3], convStride=[2, 2],
                                 batch_norm=True, step=step, ifTest=ifTest,
                                 activation=layers.ReLU,
                                 reuse=tf.compat.v1.AUTO_REUSE, name='G_DeConv8_' + str(idx))
        layers_list.append(subnet)
        subnet = layers.DeConv2D(subnet.output, convChannels=3,
                                 convKernel=[3, 3], convStride=[2, 2],
                                 batch_norm=True, step=step, ifTest=ifTest,
                                 activation=layers.ReLU,
                                 reuse=tf.compat.v1.AUTO_REUSE, name='G_SepConv3_' + str(idx))
        layers_list.append(subnet)
        subnets.append(tf.expand_dims(subnet.output, axis=-1))

    subnets = tf.concat(subnets, axis=-1)
    weights = layers.FullyConnected(tf.one_hot(targets, 1000), outputSize=num_experts,
                                    l2_constant=0.0, activation=layers.Softmax,
                                    reuse=tf.compat.v1.AUTO_REUSE, name='G_WeightsMoE')
    layers_list.append(weights)

    moe = tf.transpose(a=tf.transpose(a=subnets, perm=[1, 2, 3, 0, 4]) * weights.output, perm=[3, 0, 1, 2, 4])
    noises = (tf.nn.tanh(tf.reduce_sum(input_tensor=moe, axis=-1)) - 0.5) * NoiseRange * 2
    print('Shape of Noises: ', noises.shape)

    return noises


def create_simulator_SimpleNet(images, step, ifTest, layers_list):
    # define simulator with an architecture almost identical to SimpleNet in the paper
    net = layers.DepthwiseConv2D(preproc.normalise_images(tf.clip_by_value(images, 0, 255)), convChannels=3 * 16,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=layers.ReLU,
                                 name='DepthwiseConv3x16', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=96,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv96', dtype=tf.float32)
    layers_list.append(net)

    toadd = layers.Conv2D(net.output, convChannels=192,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='SepConv192Shortcut', dtype=tf.float32)
    layers_list.append(toadd)

    net = layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[2, 2],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv192a', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='SepConv192b', dtype=tf.float32)
    layers_list.append(net)

    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=384,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='SepConv384Shortcut', dtype=tf.float32)
    layers_list.append(toadd)

    net = layers.Activation(added, activation=layers.ReLU, name='ReLU384')
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[2, 2],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv384a', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv384b', dtype=tf.float32)
    layers_list.append(net)

    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=768,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='SepConv768Shortcut', dtype=tf.float32)
    layers_list.append(toadd)

    net = layers.Activation(added, activation=layers.ReLU, name='ReLU768')
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[2, 2],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv768a', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv768b', dtype=tf.float32)
    layers_list.append(net)

    added = toadd.output + net.output

    net = layers.Activation(added, activation=layers.ReLU, name='ReLU11024')
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv1024', dtype=tf.float32)
    layers_list.append(net)

    net = layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers_list.append(net)
    logits = layers.FullyConnected(net.output, outputSize=10, weightInit=layers.XavierInit,
                                   biasInit=layers.const_init(0.0),
                                   activation=layers.Linear,
                                   reuse=tf.compat.v1.AUTO_REUSE, name='P_FC_classes', dtype=tf.float32)
    layers_list.append(logits)

    return logits.output


def create_simulatorG_SimpleNet(images, step, ifTest):
    # define simulator with an architecture almost identical to SimpleNet in the paper
    net = layers.DepthwiseConv2D(preproc.normalise_images(tf.clip_by_value(images, 0, 255)), convChannels=3 * 16,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=layers.ReLU,
                                 name='DepthwiseConv3x16', dtype=tf.float32)
    net = layers.SepConv2D(net.output, convChannels=96,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv96', dtype=tf.float32)

    toadd = layers.Conv2D(net.output, convChannels=192,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='SepConv192Shortcut', dtype=tf.float32)

    net = layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[2, 2],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv192a', dtype=tf.float32)
    net = layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='SepConv192b', dtype=tf.float32)

    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=384,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='SepConv384Shortcut', dtype=tf.float32)

    net = layers.Activation(added, activation=layers.ReLU, name='ReLU384')
    net = layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[2, 2],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv384a', dtype=tf.float32)
    net = layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv384b', dtype=tf.float32)

    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=768,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='SepConv768Shortcut', dtype=tf.float32)

    net = layers.Activation(added, activation=layers.ReLU, name='ReLU768')
    net = layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[2, 2],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv768a', dtype=tf.float32)
    net = layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv768b', dtype=tf.float32)

    added = toadd.output + net.output

    net = layers.Activation(added, activation=layers.ReLU, name='ReLU11024')
    net = layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='SepConv1024', dtype=tf.float32)
    net = layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    logits = layers.FullyConnected(net.output, outputSize=10, weightInit=layers.XavierInit,
                                   biasInit=layers.const_init(0.0),
                                   activation=layers.Linear,
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
                         'WarmupSteps': 600,
                         'TotalSteps': 30000}


class AdvNet(nets.Net):

    def __init__(self, texture_shape, image_shape, enemy, architecture, hyper_params=None):
        nets.Net.__init__(self)

        if hyper_params is None:
            hyper_params = hyper_params_imagenet
        self._hyper_params = hyper_params
        self.renderer = AdvNet.make_uv_renderer()

        # the targeted neural network model
        self._enemy = enemy
        with self._graph.as_default():
            # Inputs
            self._std_textures = tf.compat.v1.placeholder(dtype=tf.float32,
                                                          shape=[self._hyper_params['BatchSize']] + texture_shape,
                                                          name='textures')
            self.uv_mapping = tf.compat.v1.placeholder(dtype=tf.float32,
                                                       shape=[self._hyper_params['BatchSize']] + image_shape + [2],
                                                       name='uv_mappings')
            self._labels = tf.compat.v1.placeholder(dtype=tf.int64, shape=[self._hyper_params['BatchSize']],
                                                    name='imagenet_labels')
            self._adversarial_targets = tf.compat.v1.placeholder(dtype=tf.int64,
                                                                 shape=[self._hyper_params['BatchSize']],
                                                                 name='target_labels')

            # define generator
            with tf.compat.v1.variable_scope('Generator', reuse=tf.compat.v1.AUTO_REUSE) as scope:
                self._generator = create_generator(self._std_textures, self._adversarial_targets,
                                                   self._hyper_params['NumSubnets'], self._step,
                                                   self._ifTest, self._layers)
            self._adversarial_textures = self._generator + self._std_textures

            self._print_multiplier, self._print_addend = AdvNet.get_print_error_args()
            self._photo_multiplier, self._photo_addend, self._photo_noise = AdvNet.get_photo_error_args(
                [self._hyper_params['BatchSize']] + image_shape + [3])
            self._colour = AdvNet.get_background_colour()

            self._std_images = self.get_rendering_pipeline(self._std_textures)
            self._adv_images = self.get_rendering_pipeline(self._adversarial_textures)

            # define simulator
            with tf.compat.v1.variable_scope('Simulator', reuse=tf.compat.v1.AUTO_REUSE) as scope:
                self._simulator = self.body(self._std_images, architecture, for_generator=False)
                # what is the point of this??? Why is the generator training against a different simulator, which is
                # not trained to match the target model? Why is one simulator trained on normal images, and another on
                # adversarial images?
                self._simulatorG = self.body(self._adv_images, architecture, for_generator=True)

            # define inference as hard label prediction of simulator on natural images
            self._inference = self.inference(self._simulator)
            # accuracy is how often simulator prediction matches the prediction of the target net
            self._accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self._inference, self._labels), tf.float32))

            self._layer_losses = 0
            for elem in self._layers:
                if len(elem.losses) > 0:
                    for loss in elem.losses:
                        self._layer_losses += loss

            # simulator loss matches simulator output against output of target model
            self._loss_simulator = self.loss(self._simulator, self._labels, name='lossP') + self._layer_losses
            # generator trains to produce perturbations that make the simulator produce the desired target labels
            self._loss_generator = self.loss(self._simulatorG, self._adversarial_targets, name='lossG')
            self._loss_generator += self._hyper_params['NoiseDecay'] * tf.reduce_mean(
                input_tensor=tf.norm(tensor=self._generator)) + self._layer_losses

            self._lr = tf.compat.v1.train.exponential_decay(self._hyper_params['LearningRate'],
                                                            global_step=self._step,
                                                            decay_steps=self._hyper_params['DecayAfter'],
                                                            decay_rate=self._hyper_params['DecayRate'])
            self._lr += self._hyper_params['MinLearningRate']

            print(self.summary)
            # Saver
            self._saver = tf.compat.v1.train.Saver(max_to_keep=5)

    def body(self, images, architecture, num_middle=2, for_generator=False):
        # define body of simulator
        net_output = super().body(images, architecture, num_middle, for_generator)
        logits = layers.FullyConnected(net_output, outputSize=1000, activation=layers.Linear,
                                       reuse=tf.compat.v1.AUTO_REUSE,
                                       name='P_FC_classes')
        if not for_generator:
            self._layers.append(logits)

        return logits.output

    @staticmethod
    def make_uv_renderer():
        renderer = UVRenderer(None, (299, 299))
        renderer.set_parameters(
            camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
            x_translation=(cfg.x_translation_min, cfg.x_translation_max),
            y_translation=(cfg.y_translation_min, cfg.y_translation_max)
        )

        return renderer

    def get_rendering_pipeline(self, texture):
        """Use UV mapping to create batch_seize images with both the normal and adversarial texture, then pass the
        adversarial images as input to the victim model to get logits. UV mapping is the matrix M used to transform
        texture x into the image with rendered object, as explained in the paper.

        Returns
        -------
        Tensor of shape batch_size x 1000, representing the logits obtained by passing the adversarial images as
        input to the victim model.
        """
        # create each image in batch from texture one at a time. We do this instead of all at once so that we need less
        # memory (a 12 x 2048 x 2048 x 3 tensor is 600 MB, and we would create multiple ones). We make the first image
        # outside of the loop to initialise the list of new images, and to avoid putting an if statement in the loop
        new_images = self.create_image(texture, self.uv_mapping[0])
        for i in range(cfg.batch_size - 1):
            image = self.create_image(texture, self.uv_mapping[i])
            new_images = tf.concat([new_images, image], axis=0)

        # add background colour to rendered images.
        new_images = self.add_background(new_images)

        # check if we apply random noise to simulate camera noise
        if cfg.photo_error:
            new_images = AdvNet.apply_photo_error(new_images, self._photo_multiplier, self._photo_addend, self._photo_noise)

        new_images = tf.clip_by_value(new_images, 0, 1)
        return new_images

    def create_image(self, texture, uv_mapping):
        """Create standard and adversarial images from the respective textures using the given UV mapping.

        Parameters
        ----------
        uv_mapping : numpy array
            A numpy array with shape [image_height, image_width, 2]. Represents the UV mappings for an
            image in the batch. This mappign is used to create the images from the textures.

        Returns
        -------
        tuple
            Two tensors. The first one is of shape num_new_renders x 299 x 299 x 3, representing the images of the new
            renders with the normal texture. The second is of shape num_new_renders x 299 x 299 x 3, representing the images
            of the new renders with the adversarial texture.
        """
        # check if we should add print errors, so that the adversarial texture may be used for a 3D printed object
        # and still be effective
        if cfg.print_error:
            new_texture = AdvNet.transform(texture, self._print_multiplier, self._print_addend)
        else:
            # tfa.resampler requires input to be in shape batch_size x height x width x channels, so we insert a new
            # dimension
            new_texture = tf.expand_dims(texture, axis=0)

        # tfa.image.resampler requires the first dimension of UV map to be
        # batch size, so we add an extra dimension with one element
        image_uv_map = tf.expand_dims(uv_mapping, axis=0)

        # use UV mapping to create an images corresponding to an individual render by sampling from the texture
        # Resulting tensors are of shape 1 x image_width x image_height x 3
        new_image = tfa.image.resampler(new_texture, image_uv_map)

        return new_image

    def add_background(self, images):
        """Colours the background pixels of the image with a random colour.
        """
        # compute a mask with True values for each pixel which represents the object, and False for background pixels.
        mask = tf.reduce_all(input_tensor=tf.not_equal(self.uv_mapping, 0.0), axis=3, keepdims=True)

        return AdvNet.set_background(images, mask, self._colour)

    @staticmethod
    def get_background_colour():
        return tf.random.uniform([cfg.batch_size, 1, 1, 3], cfg.background_min, cfg.background_max)

    @staticmethod
    def set_background(x, mask, colours):
        """Sets background color of an image according to a boolean mask.

        Parameters
        ----------
            x: A 4-D tensor with shape [batch_size, height, size, 3]
                The images to which a background will be added.
            mask: boolean mask with shape [batch_size, height, width, 1]
                The mask used for determining where are the background pixels. Has False for background pixels,
                True otherwise.
            colours: tensor with shape [batch_size, 1, 1, 3].
                The background colours for each image
        """
        mask = tf.tile(mask, [1, 1, 1, 3])
        inverse_mask = tf.logical_not(mask)

        return tf.cast(mask, tf.float32) * x + tf.cast(inverse_mask, tf.float32) * colours

    @staticmethod
    def get_print_error_args():
        multiplier = tf.random.uniform(
            [1, 1, 1, 3],
            cfg.channel_mult_min,
            cfg.channel_mult_max
        )
        addend = tf.random.uniform(
            [1, 1, 1, 3],
            cfg.channel_add_min,
            cfg.channel_add_max
        )

        return multiplier, addend

    @staticmethod
    def apply_photo_error(images, multiplier, addend, noise):
        images = AdvNet.transform(images, multiplier, addend)
        images += noise

        return images

    @staticmethod
    def get_photo_error_args(images_shape):
        multiplier = tf.random.uniform(
            [images_shape[0], 1, 1, 1],
            cfg.light_mult_min,
            cfg.light_mult_max
        )
        addend = tf.random.uniform(
            [images_shape[0], 1, 1, 1],
            cfg.light_add_min,
            cfg.light_add_max
        )

        gaussian_noise = tf.random.truncated_normal(
            images_shape,
            stddev=tf.random.uniform([1], maxval=cfg.stddev)
        )

        return multiplier, addend, gaussian_noise

    @staticmethod
    def transform(x, a, b):
        """Apply transform a * x + b element-wise.

         Parameters
        ----------
            x : tensor
            a : tensor
            b : tensor
        """
        return tf.add(tf.multiply(a, x), b)

    def preproc(self, images):
        # normalise images fed into the simulator
        clipped = tf.clip_by_value(images, 0, 255)
        casted = tf.cast(clipped, tf.float32)
        standardized = tf.identity(casted / 127.5 - 1.0, name='training_standardized')

        return standardized

    def inference(self, logits):
        return tf.argmax(input=logits, axis=-1, name='inference')

    def loss(self, logits, labels, name='cross_entropy'):
        net = layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output

    def train(self, data_generator, path_load=None, path_save=None):
        print("\n Begin Training: \n")
        with self._graph.as_default():
            self._step_inc = tf.compat.v1.assign(self._step, self._step + 1)

            self._varsS = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
            # define optimisers
            self._optimizerS = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._loss_simulator,
                                                                                                 var_list=self._varsS)
            self._optimizerG = self._get_generator_optimiser()

            # Initialize all
            self._sess.run(tf.compat.v1.global_variables_initializer())

            if path_load is not None:
                self.load(path_load)
            else:
                self._warm_up_simulator()

            self.evaluate(data_generator)
            self._sess.run([self._phaseTrain])
            if path_save is not None:
                self.save(path_save)

            globalStep = 0
            # main training loop
            while globalStep < self._hyper_params['TotalSteps']:
                self._sess.run(self._step_inc)

                # train simulator for a couple of steps
                for _ in range(self._hyper_params['NumPredictor']):
                    # ground truth labels are not needed for training the simulator
                    data, _, target_labels = next(data_generator)
                    # adds Random uniform noise to normal data
                    data = data + (np.random.rand(self._hyper_params['BatchSize'], 32, 32, 3) - 0.5) * 2 * NoiseRange

                    # perform one optimisation step to train simulator so it has the same predictions as the target
                    # model does on normal images with noise
                    target_model_labels = self._enemy.infer(data)
                    loss, accuracy, globalStep, _ = self._sess.run([self._loss_simulator, self._accuracy, self._step,
                                                                    self._optimizerS],
                                                                   feed_dict={self._std_textures: data,
                                                                              self._labels: target_model_labels})
                    print('\rSimulator => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; Accuracy: %.3f' % accuracy,
                          end='')

                    adversarial_images = self._sess.run(self._adversarial_textures,
                                                        feed_dict={self._std_textures: data,
                                                                   self._adversarial_targets: target_labels})
                    # perform one optimisation step to train simulator so it has the same predictions as the target
                    # model does on adversarial images
                    target_model_labels = self._enemy.infer(adversarial_images)
                    loss, accuracy, globalStep, _ = self._sess.run([self._loss_simulator, self._accuracy, self._step,
                                                                    self._optimizerS],
                                                                   feed_dict={self._std_textures: adversarial_images,
                                                                              self._labels: target_model_labels})

                    self.simulator_loss_history.append(loss)
                    self.simulator_accuracy_history.append(accuracy)
                    print('\rSimulator => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; Accuracy: %.3f' % accuracy,
                          end='')

                # train generator for a couple of steps
                for _ in range(self._hyper_params['NumGenerator']):
                    data, _, target_labels = next(data_generator)

                    loss, adversarial_images, globalStep, _ = self._sess.run([self._loss_generator,
                                                                              self._adversarial_textures,
                                                                              self._step, self._optimizerG],
                                                                             feed_dict={self._std_textures: data,
                                                                                        self._adversarial_targets: target_labels})
                    adversarial_predictions = self._enemy.infer(adversarial_images)
                    tfr = np.mean(target_labels == adversarial_predictions)
                    ufr = np.mean(target_model_labels != adversarial_predictions)

                    self.generator_loss_history.append(loss)
                    self.generator_accuracy_history.append(tfr)
                    print('\rGenerator => Step: ', globalStep,
                          '; Loss: %.3f' % loss,
                          '; TFR: %.3f' % tfr,
                          '; UFR: %.3f' % ufr,
                          end='')

                # evaluate on test every so often
                if globalStep % self._hyper_params['ValidateAfter'] == 0:
                    self.evaluate(data_generator)
                    if path_save is not None:
                        self.save(path_save)
                        np.savez("./AttackCIFAR10/training_history", self.simulator_loss_history,
                                 self.simulator_accuracy_history, self.generator_loss_history,
                                 self.generator_accuracy_history, self.test_loss_history, self.test_accuracy_history)

                    self._sess.run([self._phaseTrain])

    def _get_generator_optimiser(self):
        varsG = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

        optimizerG = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-8)
        gradientsG = optimizerG.compute_gradients(self._loss_generator, var_list=varsG)

        # clip generator optimiser gradients
        clipped_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradientsG]

        return optimizerG.apply_gradients(clipped_gradients)

    def _warm_up_simulator(self):
        # warm up simulator to match predictions of target model on clean images
        print('Warming up. ')
        for idx in range(self._hyper_params['WarmupSteps']):
            textures, _, _ = next(data_generator)
            target_model_labels = np.array(self._enemy.infer(textures))
            loss, accuracy, _ = self._sess.run([self._loss_simulator, self._accuracy, self._optimizerS],
                                               feed_dict={self._std_textures: textures,
                                                          self._labels: target_model_labels})
            print('\rSimulator => Step: ', idx - 300,
                  '; Loss: %.3f' % loss,
                  '; Accuracy: %.3f' % accuracy,
                  end='')

        # evaluate warmed up simulator on test data
        warmup_accuracy = 0.0
        for idx in range(50):
            textures, _, _ = next(data_generator)
            target_model_labels = np.array(self._enemy.infer(textures))
            loss, accuracy = self._sess.run([self._loss_simulator, self._accuracy],
                                            feed_dict={self._std_textures: textures,
                                                       self._labels: target_model_labels})
            warmup_accuracy += accuracy
        warmup_accuracy = warmup_accuracy / 50
        print('\nWarmup Accuracy: ', warmup_accuracy)

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
                                                      feed_dict={self._std_textures: data,
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
                                                  feed_dict={self._std_textures: data,
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
                           feed_dict={self._std_textures: tmpdata,
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

    net = AdvNet([2048, 2048, 3], [299, 299], enemy, "SimpleNet")
    data_generator = data.get_adversarial_data_generators(batch_size=hyper_params_imagenet['BatchSize'])

    net.train(data_generator, path_save='./AttackCIFAR10/netcifar10.ckpt')
    net.plot_training_history("Adversarial CIFAR10", net._hyper_params['ValidateAfter'])
