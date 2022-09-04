import tensorflow as tf
from layers import conv2d_bn, sep_conv2d_bn, depthwise_conv2d_bn
import numpy as np
import os
import matplotlib.pyplot as plt

import preproc


relu = tf.keras.activations.relu


class Net:

    def __init__(self):
        self._body = None

        self.generator_loss_history = []
        self.generator_l2_loss_history = []
        self.generator_tfr_history = []

        self.simulator_loss_history = []
        self.simulator_accuracy_history = []

        self.test_loss_history = []
        self.test_accuracy_history = []

    def create_simulator(self, architecture, num_middle=2):
        """
        Defines the body of the NN and adds all layers to the layers list of the NN.

        Parameters
        ----------
        images : Tensor
            Tensor representing the minibatch of images that the NN works on. They must not be normalised, and have
            values between 0 and 255.
        architecture : str
            String naming the architecture to use for the simulator of this model.
        num_middle : int
            Number used if architecture is Xception to control how deep the network is.
        for_generator : bool
            If this simulator is duplicate, with the same weights, to have the generator output as input. Used for the
            generator loss.

        Returns
        ----------
        outputs
            Output of NN as a Tensor.
        """

        if architecture == "SimpleNet":
            images, logits = get_Simple_Net()
        elif architecture == "SmallNet":
            net = SmallNet(standardized, self._step, self._ifTest, layers_list)
        elif architecture == "ConcatNet":
            net = ConcatNet(standardized, self._step, self._ifTest, layers_list)
        elif architecture == "Xception":
            net = Xception(standardized, self._step, self._ifTest, layers_list, numMiddle=num_middle)
        else:
            raise ValueError("Invalid simulator architecture argument!")

        return tf.keras.Model(inputs=images, outputs=logits)

    def train(self, data_generator):
        """
        Trains network according the hyper params of the Net subclass.

        Parameters
        ----------
        training_data_generator : generator
            Generator which returns each step a tuple with two tensors: the first is the mini-batch of training images,
            and the second is a list of their coresponding hard labels.
        path_load : string
            Path to checkpoint with weights of pre-trained model that we want to further train.
        path_save : string
            Path to where we want to save a checkpoint with the current weights of the model.
        """
        pass

    def evaluate(self, test_data_generator):
        """
        Evaluates trained (or in training) model across several minibatches from the test set. The number of batches is
        a hyper param of the Net subclass.

        Parameters
        ----------
        test_data_generator : generator
            Generator which returns each step a tuple with two tensors: the first is the mini-batch of test images, and
            the second is a list of their coresponding hard labels.
        path : string
            Path to checkpoint with weights of pre-trained model that we want to evaluate
        """
        pass

    def save(self, path):
        self._saver.save(self._sess, path, global_step=self._step)

    def load(self, path):
        """
        Restores model variables from a file at the given path.
        Parameters
        ----------
        path : String
            Path to file containing saved variables.
        """
        self._saver.restore(self._sess, path)
        self.load_training_history("./AttackCIFAR10/training_history")

    def load_training_history(self, path):
        assert type(path) is str

        if os.path.exists(path):
            array_dict = np.load(path)

            self.simulator_loss_history = array_dict['arr_0']
            self.simulator_accuracy_history = array_dict['arr_1']
            self.generator_loss_history = array_dict['arr_2']
            self.generator_tfr_history = array_dict['arr_3']
            self.test_loss_history = array_dict['arr_4']
            self.test_accuracy_history = array_dict['arr_5']
            print("Training history restored.")

    def plot_training_history(self, model, test_after):
        plt.plot(self.simulator_loss_history, label="Simulator")
        plt.plot(self.generator_loss_history, label="Generator")
        test_steps = list(range(0, len(self.simulator_loss_history) + 1, test_after))
        plt.plot(test_steps, self.test_loss_history, label="Generator Test")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("{} loss history".format(model))
        plt.legend()
        plt.show()

        plt.plot(self.simulator_accuracy_history, label="Simulator")
        plt.plot(self.generator_tfr_history, label="Generator")
        test_steps = list(range(0, len(self.simulator_accuracy_history) + 1, test_after))
        plt.plot(test_steps, self.test_accuracy_history, label="Generator Test")
        plt.xlabel("Steps")
        plt.ylabel("TFR")
        plt.title("{} TFR history".format(model))
        plt.legend()
        plt.show()


# has two fewer layers compared to diagram in paper, misses last two conv 128 layers
def SmallNet(standardized, step, ifTest, layer_list):
    net = layers.Conv2D(standardized, convChannels=64,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv1', dtype=tf.float32)
    layer_list.append(net)
    net = layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv2', dtype=tf.float32)
    layer_list.append(net)
    net = layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[2, 2],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv3', dtype=tf.float32)
    layer_list.append(net)
    toadd = net.output
    net = layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv4a', dtype=tf.float32)
    layer_list.append(net)
    added = toadd + net.output
    net = layers.Conv2D(added, convChannels=128,
                        convKernel=[3, 3], convStride=[2, 2],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv5', dtype=tf.float32)
    layer_list.append(net)
    toadd = net.output
    net = layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv6a', dtype=tf.float32)
    layer_list.append(net)
    added = toadd + net.output
    net = layers.Conv2D(added, convChannels=128,
                        convKernel=[3, 3], convStride=[2, 2],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv7', dtype=tf.float32)
    layer_list.append(net)
    toadd = net.output
    net = layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv8a', dtype=tf.float32)
    layer_list.append(net)
    added = toadd + net.output
    net = layers.Conv2D(added, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv9', dtype=tf.float32)
    layer_list.append(net)
    net = layers.Conv2D(net.output, convChannels=64,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Conv10', dtype=tf.float32)
    layer_list.append(net)

    return net


def get_Simple_Net():
    # initial three layers in entry flow
    images = tf.keras.layers.Input(shape=(299, 299, 3), dtype=tf.float32)
    x = conv2d_bn(images, filters=24, kernel_size=3, strides=2, activation=relu)
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
    x = tf.keras.layers.Dense(units=1000, activation=None)(x)

    return images, x


def ConcatNet(standardized, step, ifTest, layers_list, numMiddle=2):
    # why does this not have activation?
    net = layers.DepthwiseConv2D(standardized, convChannels=48,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 name='DepthwiseConv3x16', dtype=tf.float32)
    layers_list.append(net)

    # this layer does not show up in paper diagram
    toconcat = layers.Conv2D(net.output, convChannels=48,
                             convKernel=[3, 3], convStride=[1, 1],
                             convInit=layers.XavierInit, convPadding='SAME',
                             biasInit=layers.const_init(0.0),
                             batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=layers.ReLU,
                             name='Stage1_Conv_48a', dtype=tf.float32)
    layers_list.append(toconcat)

    net = layers.Conv2D(toconcat.output, convChannels=96,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Stage1_Conv1x1_96', dtype=tf.float32)
    layers_list.append(net)
    net = layers.DepthwiseConv2D(net.output, convChannels=96,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=layers.ReLU,
                                 name='Stage1_DepthwiseConv96', dtype=tf.float32)
    layers_list.append(net)
    net = layers.Conv2D(net.output, convChannels=48,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.Linear,
                        name='Stage1_Conv1x1_48b', dtype=tf.float32)
    layers_list.append(net)

    concated = tf.concat([toconcat.output, net.output], axis=3)

    toconcat = layers.Conv2D(concated, convChannels=96,
                             convKernel=[3, 3], convStride=[1, 1],
                             convInit=layers.XavierInit, convPadding='SAME',
                             biasInit=layers.const_init(0.0),
                             batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=layers.ReLU,
                             name='Stage2_Conv_96a', dtype=tf.float32)
    layers_list.append(toconcat)

    net = layers.Conv2D(toconcat.output, convChannels=192,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Stage2_Conv1x1_192', dtype=tf.float32)
    layers_list.append(net)
    net = layers.DepthwiseConv2D(net.output, convChannels=192,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=layers.ReLU,
                                 name='Stage2_DepthwiseConv192', dtype=tf.float32)
    layers_list.append(net)
    net = layers.Conv2D(net.output, convChannels=96,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.Linear,
                        name='Stage2_Conv1x1_96b', dtype=tf.float32)
    layers_list.append(net)

    concated = tf.concat([toconcat.output, net.output], axis=3)

    toconcat = layers.Conv2D(concated, convChannels=192,
                             convKernel=[3, 3], convStride=[1, 1],
                             convInit=layers.XavierInit, convPadding='SAME',
                             biasInit=layers.const_init(0.0),
                             batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=layers.ReLU,
                             pool=True, poolSize=[3, 3], poolStride=[2, 2],
                             poolType=layers.MaxPool, poolPadding='SAME',
                             name='Stage3_Conv_192a', dtype=tf.float32)
    layers_list.append(toconcat)

    net = layers.Conv2D(toconcat.output, convChannels=384,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Stage3_Conv1x1_384', dtype=tf.float32)
    layers_list.append(net)
    net = layers.DepthwiseConv2D(net.output, convChannels=384,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=layers.ReLU,
                                 name='Stage3_DepthwiseConv384', dtype=tf.float32)
    layers_list.append(net)
    net = layers.Conv2D(net.output, convChannels=192,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.Linear,
                        name='Stage3_Conv1x1_192b', dtype=tf.float32)
    layers_list.append(net)

    concated = tf.concat([toconcat.output, net.output], axis=3)

    toconcat = layers.Conv2D(concated, convChannels=384,
                             convKernel=[3, 3], convStride=[1, 1],
                             convInit=layers.XavierInit, convPadding='SAME',
                             biasInit=layers.const_init(0.0),
                             batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=layers.ReLU,
                             pool=True, poolSize=[3, 3], poolStride=[2, 2],
                             poolType=layers.MaxPool, poolPadding='SAME',
                             name='Stage4_Conv_384a', dtype=tf.float32)
    layers_list.append(toconcat)

    net = layers.Conv2D(toconcat.output, convChannels=768,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Stage4_Conv1x1_768', dtype=tf.float32)
    layers_list.append(net)
    net = layers.DepthwiseConv2D(net.output, convChannels=768,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=layers.ReLU,
                                 name='Stage4_DepthwiseConv768', dtype=tf.float32)
    layers_list.append(net)
    net = layers.Conv2D(net.output, convChannels=384,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.Linear,
                        name='Stage4_Conv1x1_384b', dtype=tf.float32)
    layers_list.append(net)

    concated = tf.concat([toconcat.output, net.output], axis=3)
    # again, this layer does not show up in diagram
    toadd = layers.Conv2D(concated, convChannels=768,
                          convKernel=[3, 3], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=layers.Linear,
                          name='SepConv768Toadd', dtype=tf.float32)
    layers_list.append(toadd)
    conved = toadd.output

    for idx in range(numMiddle):
        net = layers.Activation(conved, layers.ReLU, name='ActMiddle' + str(idx) + '_1')
        layers_list.append(net)
        net = layers.SepConv2D(net.output, convChannels=768,
                               convKernel=[3, 3], convStride=[1, 1],
                               convInit=layers.XavierInit, convPadding='SAME',
                               biasInit=layers.const_init(0.0),
                               batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle' + str(idx) + '_1', dtype=tf.float32)
        layers_list.append(net)
        net = layers.Activation(net.output, layers.ReLU, name='ReLUMiddle' + str(idx) + '_2')
        layers_list.append(net)
        net = layers.SepConv2D(net.output, convChannels=768,
                               convKernel=[3, 3], convStride=[1, 1],
                               convInit=layers.XavierInit, convPadding='SAME',
                               biasInit=layers.const_init(0.0),
                               batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle' + str(idx) + '_2', dtype=tf.float32)
        layers_list.append(net)
        net = layers.Activation(net.output, layers.ReLU, name='ReLUMiddle' + str(idx) + '_3')
        layers_list.append(net)
        net = layers.SepConv2D(net.output, convChannels=768,
                               convKernel=[3, 3], convStride=[1, 1],
                               convInit=layers.XavierInit, convPadding='SAME',
                               biasInit=layers.const_init(0.0),
                               batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle' + str(idx) + '_3', dtype=tf.float32)
        layers_list.append(net)
        conved = net.output + conved

    # this skip connection does not show up in paper
    toadd = layers.Conv2D(conved, convChannels=1536,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=layers.MaxPool, poolPadding='SAME',
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers_list.append(toadd)

    net = layers.Activation(conved, layers.ReLU, name='ActExit768_1')
    layers_list.append(net)

    # this does not show up in diagram in paper
    toconcat = layers.Conv2D(net.output, convChannels=768,
                             convKernel=[3, 3], convStride=[1, 1],
                             convInit=layers.XavierInit, convPadding='SAME',
                             biasInit=layers.const_init(0.0),
                             batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=layers.ReLU,
                             pool=True, poolSize=[3, 3], poolStride=[2, 2],
                             poolType=layers.MaxPool, poolPadding='SAME',
                             name='ConvExit768_1', dtype=tf.float32)
    layers_list.append(toconcat)

    net = layers.Conv2D(toconcat.output, convChannels=1536,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='Exit_Conv1x1_1536', dtype=tf.float32)
    layers_list.append(net)
    net = layers.DepthwiseConv2D(net.output, convChannels=1536,
                                 convKernel=[3, 3], convStride=[1, 1],
                                 convInit=layers.XavierInit, convPadding='SAME',
                                 biasInit=layers.const_init(0.0),
                                 batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=layers.ReLU,
                                 name='Exit_DepthwiseConv1536', dtype=tf.float32)
    layers_list.append(net)
    net = layers.Conv2D(net.output, convChannels=768,
                        convKernel=[1, 1], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.Linear,
                        name='Exit_Conv1x1_768b', dtype=tf.float32)
    layers_list.append(net)

    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output

    # does not show up in diagram in paper
    net = layers.SepConv2D(added, convChannels=2048,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='ConvExit2048_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers_list.append(net)

    return net


def Xception(standardized, step, ifTest, layers_list, numMiddle=8):
    net = layers.Conv2D(standardized, convChannels=32,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='ConvEntry32_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.Conv2D(net.output, convChannels=64,
                        convKernel=[3, 3], convStride=[1, 1],
                        convInit=layers.XavierInit, convPadding='SAME',
                        biasInit=layers.const_init(0.0),
                        batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=layers.ReLU,
                        name='ConvEntry64_1', dtype=tf.float32)
    layers_list.append(net)

    toadd = layers.Conv2D(net.output, convChannels=128,
                          convKernel=[1, 1], convStride=[1, 1],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers_list.append(toadd)

    net = layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='ConvEntry128_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='ConvEntry128_2', dtype=tf.float32)
    layers_list.append(net)

    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=256,
                          convKernel=[1, 1], convStride=[2, 2],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers_list.append(toadd)

    acted = layers.Activation(added, layers.ReLU, name='ReLUEntry256_0')
    layers_list.append(acted)
    net = layers.SepConv2D(acted.output, convChannels=256,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='ConvEntry256_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=256,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           pool=True, poolSize=[3, 3], poolStride=[2, 2],
                           poolType=layers.MaxPool, poolPadding='SAME',
                           name='ConvEntry256_2', dtype=tf.float32)
    layers_list.append(net)
    added = toadd.output + net.output

    toadd = layers.Conv2D(added, convChannels=728,
                          convKernel=[1, 1], convStride=[2, 2],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers_list.append(toadd)

    acted = layers.Activation(added, layers.ReLU, name='ReLUEntry728_0')
    layers_list.append(acted)
    net = layers.SepConv2D(acted.output, convChannels=728,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='ConvEntry728_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=728,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           pool=True, poolSize=[3, 3], poolStride=[2, 2],
                           poolType=layers.MaxPool, poolPadding='SAME',
                           name='ConvEntry728_2', dtype=tf.float32)
    layers_list.append(net)
    added = toadd.output + net.output
    conved = added

    for idx in range(numMiddle):
        net = layers.Activation(conved, layers.ReLU, name='ActMiddle' + str(idx) + '_1')
        layers_list.append(net)
        net = layers.SepConv2D(net.output, convChannels=728,
                               convKernel=[3, 3], convStride=[1, 1],
                               convInit=layers.XavierInit, convPadding='SAME',
                               biasInit=layers.const_init(0.0),
                               batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle' + str(idx) + '_1', dtype=tf.float32)
        layers_list.append(net)
        net = layers.Activation(net.output, layers.ReLU, name='ReLUMiddle' + str(idx) + '_2')
        layers_list.append(net)
        net = layers.SepConv2D(net.output, convChannels=728,
                               convKernel=[3, 3], convStride=[1, 1],
                               convInit=layers.XavierInit, convPadding='SAME',
                               biasInit=layers.const_init(0.0),
                               batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle' + str(idx) + '_2', dtype=tf.float32)
        layers_list.append(net)
        net = layers.Activation(net.output, layers.ReLU, name='ReLUMiddle' + str(idx) + '_3')
        layers_list.append(net)
        net = layers.SepConv2D(net.output, convChannels=728,
                               convKernel=[3, 3], convStride=[1, 1],
                               convInit=layers.XavierInit, convPadding='SAME',
                               biasInit=layers.const_init(0.0),
                               batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle' + str(idx) + '_3', dtype=tf.float32)
        layers_list.append(net)
        conved = net.output + conved

    toadd = layers.Conv2D(conved, convChannels=1024,
                          convKernel=[1, 1], convStride=[2, 2],
                          convInit=layers.XavierInit, convPadding='SAME',
                          biasInit=layers.const_init(0.0),
                          batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers_list.append(toadd)

    net = layers.Activation(conved, layers.ReLU, name='ActExit728_1')
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=728,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='ConvExit728_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           pool=True, poolSize=[3, 3], poolStride=[2, 2],
                           poolType=layers.MaxPool, poolPadding='SAME',
                           name='ConvExit1024_1', dtype=tf.float32)
    layers_list.append(net)
    added = toadd.output + net.output

    net = layers.SepConv2D(added, convChannels=1536,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='ConvExit1536_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.SepConv2D(net.output, convChannels=2048,
                           convKernel=[3, 3], convStride=[1, 1],
                           convInit=layers.XavierInit, convPadding='SAME',
                           biasInit=layers.const_init(0.0),
                           batch_norm=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=layers.ReLU,
                           name='ConvExit2048_1', dtype=tf.float32)
    layers_list.append(net)
    net = layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers_list.append(net)

    return net
