import tensorflow as tf
# gpu = tf.config.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_logical_device_configuration(
#     gpu,
#     [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])

import matplotlib.pyplot as plt
import random
import numpy as np

import data
import layers
import nets
import encoders
import preproc
import differentiable_rendering as diff_rendering

NoiseRange = 10.0
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits


def create_encoder(textures, step, ifTest, layers_list):
    return encoders.get_Simple_Net_encoder(textures, step, ifTest, layers_list, name_prefix="G_")


def create_generator(textures, targets, num_experts, step, ifTest, layers_list):
    textures = preproc.normalise_images_for_net(textures)
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
    net = layers.DepthwiseConv2D(preproc.normalise_images_for_net(tf.clip_by_value(images, 0, 255)), convChannels=3 * 16,
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
    net = layers.DepthwiseConv2D(preproc.normalise_images_for_net(tf.clip_by_value(images, 0, 255)), convChannels=3 * 16,
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


hyper_params_imagenet = {'BatchSize': 6,
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
                         'WarmupSteps': 100,
                         'TotalSteps': 30000}


class AdvNet(nets.Net):

    def __init__(self, image_shape, architecture, hyper_params=None):
        nets.Net.__init__(self)

        if hyper_params is None:
            hyper_params = hyper_params_imagenet
        self._hyper_params = hyper_params
        self.image_shape = image_shape

        self.enemy = tf.keras.applications.inception_v3.InceptionV3(
            include_top=True,
            weights='imagenet',
            classifier_activation=None
        )
        self.enemy.trainable = False

        # Inputs
        self.adv_images = tf.zeros(shape=[self._hyper_params['BatchSize']] + self.image_shape + [3], dtype=tf.float32,
                                   name="adversarial images")

        # define generator
        self.generator = create_generator(self._hyper_params['NumSubnets'])
        # define simulator
        self.simulator = self.create_simulator(architecture)

        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self._hyper_params['LearningRate'], decay_steps=self._hyper_params['DecayAfter'],
            decay_rate=self._hyper_params['DecayRate'])

        self.generator_optimiser = tf.keras.optimizers.Adam(learning_rate_schedule, epsilon=1e-8)
        self.simulator_optimiser = tf.keras.optimizers.Adam(learning_rate_schedule, epsilon=1e-8)

    def create_simulator(self, images, architecture, num_middle=2):
        # define body of simulator
        net_output = super().body(images, architecture, num_middle)
        logits = layers.FullyConnected(net_output, outputSize=1000, activation=layers.Linear,
                                       reuse=tf.compat.v1.AUTO_REUSE,
                                       name='S_FC_classes')

        return logits.output

    # define inference as hard label prediction of simulator on natural images
    @staticmethod
    def inference(logits):
        """
        Computes hard label prediction (label is one categorical value, not a vector of probabilities for each class.)

        Parameters
        ----------
        logits : Tensor
            Tensor representing the output of the NN with one minibatch as input.

        Returns
        ----------
        predictions
            A tensor with the label prediction for every sample in the minibatch.
        """
        return tf.argmax(input=logits, axis=-1, name='inference')

    @staticmethod
    def accuracy(predictions, labels):
        return tf.reduce_mean(input_tensor=tf.cast(tf.equal(predictions, labels), dtype=tf.float32))

    def train(self, data_generator):
        print("\n Begin Training: \n")

        self.warm_up_simulator()
        self.evaluate(data_generator)

        globalStep = 0
        # main training loop
        while globalStep < self._hyper_params['TotalSteps']:
            # train simulator for a couple of steps
            for _ in range(self._hyper_params['NumSimulator']):
                # ground truth labels are not needed for training the simulator
                textures, uv_maps, _, target_labels = next(data_generator)

                # perform one optimisation step to train simulator so it has the same predictions as the target
                # model does on adversarial images
                adversarial_textures = self.generator(textures) + textures
                print('\rSimulator => Step: {}'.format(globalStep), end='')
                self.simulator_training_step(adversarial_textures, uv_maps)

                # adds Random uniform noise to normal data
                textures += (np.random.rand(self._hyper_params['BatchSize'], 32, 32, 3) - 0.5) * 2 * NoiseRange
                # perform one optimisation step to train simulator so it has the same predictions as the target
                # model does on normal images with noise
                print('\rSimulator => Step: {}'.format(globalStep), end='')
                self.simulator_training_step(textures, uv_maps)

            # train generator for a couple of steps
            for _ in range(self._hyper_params['NumGenerator']):
                textures, uv_maps, true_labels, target_labels = next(data_generator)
                self.generator_training_step(textures, uv_maps, target_labels)

                enemy_labels = AdvNet.inference(self.enemy(self.adv_images))
                tfr = np.mean(target_labels.numpy() == enemy_labels.numpy())
                ufr = np.mean(true_labels != enemy_labels.numpy())
                self.generator_tfr_history.append(tfr)
                print('; TFR: %.3f' % tfr, '; UFR: %.3f' % ufr, end='')

            # evaluate on test every so often
            if globalStep % self._hyper_params['ValidateAfter'] == 0:
                self.evaluate(data_generator)

    def warm_up_simulator(self):
        # warm up simulator to match predictions of target model on clean images
        print('Warming up. ')
        for i in range(self._hyper_params['WarmupSteps']):
            textures, uv_maps, _, _ = next(data_generator)
            self.simulator_training_step(textures, uv_maps)

            print('\rSimulator => Step: ', i - self._hyper_params['WarmupSteps'], end='')

        # evaluate warmed up simulator on test data
        warmup_accuracy = 0.0
        print("Evaluating warmed up simulator:")
        for i in range(50):
            textures, uv_maps, true_labels, _ = next(data_generator)
            warmup_accuracy += self.warm_up_evaluation(textures, uv_maps)

        warmup_accuracy = warmup_accuracy / 50
        print('\nAverage Warmup Accuracy: ', warmup_accuracy)

    def simulator_training_step(self, textures, uv_maps):
        with tf.GradientTape() as simulator_tape:
            print_error_params = diff_rendering.get_print_error_args()
            photo_error_params = diff_rendering.get_photo_error_args([self._hyper_params['BatchSize']] +
                                                                     self.image_shape + [3])
            background_colours = diff_rendering.get_background_colours()

            images = diff_rendering.render(textures, uv_maps, print_error_params, photo_error_params,
                                           background_colours)

            sim_loss = self.simulator_loss(images)

        simulator_gradients = simulator_tape.gradient(sim_loss, self.simulator.trainable_variables)
        self.simulator_optimiser.apply_gradients(zip(simulator_gradients, self.simulator.trainable_variables))

    def generator_training_step(self, std_textures, uv_maps, target_labels):
        with tf.GradientTape() as generator_tape:
            gen_loss = self.generator_loss(std_textures, uv_maps, target_labels)

        generator_gradients = generator_tape.gradient(gen_loss, self.generator.trainable_variables)
        # clip generator gradients
        generator_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in generator_gradients]

        self.generator_optimiser.apply_gradients(zip(generator_gradients), self.generator.trainable_variables)

    # generator trains to produce perturbations that make the simulator produce the desired target label
    def generator_loss(self, textures, uv_maps, target_labels):
        adversarial_noises = self.generator(textures)
        adversarial_textures = textures + adversarial_noises
        adversarial_textures = diff_rendering.general_normalisation(adversarial_textures)

        print_error_params = diff_rendering.get_print_error_args()
        photo_error_params = diff_rendering.get_photo_error_args([self._hyper_params['BatchSize']] + self.image_shape + [3])
        background_colour = diff_rendering.get_background_colours()
        self.adv_images = diff_rendering.render(adversarial_textures, uv_maps, print_error_params,
                                                photo_error_params, background_colour)

        simulator_logits = self.simulator(self.adv_images)

        main_loss = cross_entropy(logits=simulator_logits, labels=target_labels)
        main_loss = tf.reduce_mean(main_loss)

        l2_penalty = self._hyper_params['NoiseDecay'] * tf.reduce_mean(input_tensor=tf.norm(tensor=adversarial_noises))

        self.generator_loss_history.append(main_loss.numpy())
        self.generator_l2_loss_history.append(l2_penalty.numpy())

        loss = main_loss + l2_penalty
        print("; Loss: %.3f" % loss.numpy(), end='')

        return main_loss + l2_penalty

    def simulator_loss(self, images):
        simulator_logits = self.simulator(images)
        enemy_model_labels = AdvNet.inference(self.enemy(images))

        loss = cross_entropy(logits=simulator_logits, labels=enemy_model_labels)
        loss = tf.reduce_mean(loss)

        self.simulator_loss_history.append(loss.numpy())
        print("; Loss: %.3f" % loss.numpy(), end='')

        accuracy = AdvNet.accuracy(AdvNet.inference(simulator_logits), enemy_model_labels)
        self.simulator_accuracy_history.append(accuracy.numpy())
        print("; Accuracy: %.3f" % accuracy.numpy(), end='')
        return loss

    def warm_up_evaluation(self, textures, uv_maps):
        print_error_params = diff_rendering.get_print_error_args()
        photo_error_params = diff_rendering.get_photo_error_args([self._hyper_params['BatchSize']] +
                                                                 self.image_shape + [3])
        background_colours = diff_rendering.get_background_colours()

        images = diff_rendering.render(textures, uv_maps, print_error_params, photo_error_params,
                                       background_colours)

        simulator_logits = self.simulator(images)
        enemy_model_labels = AdvNet.inference(self.enemy(images))

        accuracy = AdvNet.accuracy(AdvNet.inference(simulator_logits), enemy_model_labels)
        print("\rAccuracy: %.3f" % accuracy.numpy(), end='')
        return accuracy

    def evaluate(self, test_data_generator):
        total_loss = 0.0
        total_tfr = 0.0
        total_ufr = 0.0

        for _ in range(self._hyper_params['TestSteps']):
            textures, uv_maps, true_labels, target_labels = next(test_data_generator)
            # create adv image by adding the generated adversarial noise
            textures += self.generator(textures)
            textures = diff_rendering.general_normalisation(textures)

            # use adversarial textures to render adversarial images
            print_error_params = diff_rendering.get_print_error_args()
            photo_error_params = diff_rendering.get_photo_error_args([self._hyper_params['BatchSize']] +
                                                                     self.image_shape + [3])
            background_colours = diff_rendering.get_background_colours()
            images = diff_rendering.render(textures, uv_maps, print_error_params, photo_error_params,
                                           background_colours)

            # evaluate adversarial images on target model
            enemy_model_logits = self.enemy(images)
            enemy_model_labels = AdvNet.inference(enemy_model_logits)

            main_loss = cross_entropy(logits=enemy_model_logits, labels=target_labels)
            main_loss = tf.reduce_mean(main_loss)

            tfr = np.mean(target_labels.numpy() == enemy_model_labels.numpy())
            ufr = np.mean(true_labels.numpy() != enemy_model_labels.numpy())

            total_loss += main_loss
            total_tfr += tfr
            total_ufr += ufr

        total_loss /= self._hyper_params['TestSteps']
        total_tfr /= self._hyper_params['TestSteps']
        total_ufr /= self._hyper_params['TestSteps']

        self.test_loss_history.append(total_loss)
        self.test_accuracy_history.append(total_tfr)
        print('\nTest: Loss: ', total_loss, '; TFR: ', total_tfr, '; UFR: ', total_ufr)

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
    gpu = tf.config.list_physical_devices('GPU')[0]
    print(tf.test.is_gpu_available())
    print(tf.test.gpu_device_name())

    net = AdvNet([2048, 2048, 3], [299, 299], "SimpleNet")
    data_generator = data.get_adversarial_data_generators(batch_size=hyper_params_imagenet['BatchSize'])

    net.train(data_generator)
    net.plot_training_history("Adversarial CIFAR10", net._hyper_params['ValidateAfter'])
