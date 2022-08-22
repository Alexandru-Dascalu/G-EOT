import tensorflow as tf
import Layers
import numpy as np
import os
import matplotlib.pyplot as plt

wd = 4e-5


class Net:

    def __init__(self):
        self._layers = []
        self._variables = []
        self._body = None
        self._inference = None
        self._loss = None
        self._saver = None

        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)

        with self._graph.as_default():
            # variable to keep check if network is being tested or trained
            self._ifTest = tf.Variable(False, name='ifTest', trainable=False, dtype=tf.bool)
            # define operations to set ifTest variable
            self._phaseTrain = tf.compat.v1.assign(self._ifTest, False)
            self._phaseTest = tf.compat.v1.assign(self._ifTest, True)

            self._step = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)

        self.generator_loss_history = []
        self.generator_accuracy_history = []
        self.simulator_loss_history = []
        self.simulator_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

    def body(self, images, architecture, num_middle=2, for_generator=False):
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
        # preprocess images
        standardized = self.preproc(images)

        # when we duplicate the simulator, with the same weights, and tie it to the generator output, we do not want to
        # double the regularisation losses for each layer. Therefore, we pass in an empty list, so that layers are not
        # added a second time to self._layers. If subclass is a target model, this argument should always be False
        if for_generator:
            layers = []
        else:
            layers = self._layers

        if architecture == "SimpleNet":
            net = SimpleNet(standardized, self._step, self._ifTest, layers)
        elif architecture == "SmallNet":
            net = SmallNet(standardized, self._step, self._ifTest, layers)
        elif architecture == "ConcatNet":
            net = ConcatNet(standardized, self._step, self._ifTest, layers)
        elif architecture == "Xception":
            net = Xception(standardized, self._step, self._ifTest, layers, numMiddle=num_middle)
        else:
            raise ValueError("Invalid simulator architecture argument!")

        return net.output


    def preproc(self, images):
        pass

    def inference(self, logits):
        """
        Computes hard label prediction (label is one categrical value, not a vector of probabilities for each class.)

        Parameters
        ----------
        logits : Tensor
            Tensor representing the output of the NN with one minibatch as input.

        Returns
        ----------
        predictions
            A tensor with the label prediction for every sample in the minibatch.
        """
        pass

    def loss(self, logits, labels, name):
        """
        Computes loss function of the NN.

        Parameters
        ----------
        logits : Tensor
            Tensor representing the output of the NN with one minibatch as input.
        labels : Tensor
            Tensor representing labels of each sample in minibatch. Are they always soft or hard labels?
        name : string
            Name of loss function.

        Returns
        ----------
        predictions
            A tensor with the label prediction for every sample in the minibatch.
        """
        pass

    def train(self, training_data_generator, test_data_generator, path_load=None, path_save=None):
        """
        Trains network according the the hyper params of the Net subclass.

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

    def evaluate(self, test_data_generator, path=None):
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
            self.generator_accuracy_history = array_dict['arr_3']
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
        plt.plot(self.generator_accuracy_history, label="Generator")
        test_steps = list(range(0, len(self.simulator_accuracy_history) + 1, test_after))
        plt.plot(test_steps, self.test_accuracy_history, label="Generator Test")
        plt.xlabel("Steps")
        plt.ylabel("TFR")
        plt.title("{} TFR history".format(model))
        plt.legend()
        plt.show()

    @property
    def summary(self):
        summs = []
        summs.append("=>Network Summary: ")
        for elem in self._layers:
            summs.append(elem.summary)
        summs.append("<=Network Summary: ")
        return "\n\n".join(summs)


# has two fewer layers compared to diagram in paper, misses last two conv 128 layers
def SmallNet(standardized, step, ifTest, layers):
    net = Layers.Conv2D(standardized, convChannels=64,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv2', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[2, 2], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv3', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv4a', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=128,
                        convKernel=[3, 3], convStride=[2, 2], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv5', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv6a', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=128,
                        convKernel=[3, 3], convStride=[2, 2], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv7', dtype=tf.float32)
    layers.append(net)
    toadd = net.output
    net = Layers.Conv2D(net.output, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv8a', dtype=tf.float32)
    layers.append(net)
    added = toadd + net.output
    net = Layers.Conv2D(added, convChannels=128,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv9', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Conv10', dtype=tf.float32)
    layers.append(net)
    flattened = tf.reshape(net.output, [-1, net.output.shape[1]*net.output.shape[2]*net.output.shape[3]])
    net = Layers.FullyConnected(flattened, outputSize=1024, weightInit=Layers.XavierInit, wd=wd,
                                biasInit=Layers.ConstInit(0.0),
                                activation=Layers.ReLU,
                                name='FC1', dtype=tf.float32)
    layers.append(net)
    net = Layers.BatchNorm(net.output, step, ifTest, epsilon=1e-5, name='BatchNormFC1', dtype=tf.float32)
    layers.append(net)
    
    return net

def SimpleNet(standardized, step, ifTest, layers):
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=96,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv96', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=192,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv192Shortcut', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv192a', dtype=tf.float32)
    layers.append(net)
    # why does this not have activation?
    net = Layers.SepConv2D(net.output, convChannels=192,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='SepConv192b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=384,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv384Shortcut', dtype=tf.float32)
    layers.append(toadd)

    # why activate this again?
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU384')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv384a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=384,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv384b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=768,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.ReLU,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='SepConv768Shortcut', dtype=tf.float32)
    layers.append(toadd)

    # why activate this again?
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU768')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[2, 2], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv768a', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=768,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv768b', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output

    # why activate this? both toadd and net had RELU activation
    net = Layers.Activation(added, activation=Layers.ReLU, name='ReLU11024')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='SepConv1024', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    return net


def ConcatNet(standardized, step, ifTest, layers, numMiddle=2):
    # why does this not have activation?
    net = Layers.DepthwiseConv2D(standardized, convChannels=3*16,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 name='DepthwiseConv3x16', dtype=tf.float32)
    layers.append(net)

    # this layer does not show up in paper diagram
    toconcat = Layers.Conv2D(net.output, convChannels=48,
                             convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                             convInit=Layers.XavierInit, convPadding='SAME',
                             biasInit=Layers.ConstInit(0.0),
                             batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=Layers.ReLU,
                             name='Stage1_Conv_48a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=96,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Stage1_Conv1x1_96', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=96,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='Stage1_DepthwiseConv96', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=48,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.Linear,
                        name='Stage1_Conv1x1_48b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=96,
                             convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                             convInit=Layers.XavierInit, convPadding='SAME',
                             biasInit=Layers.ConstInit(0.0),
                             batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=Layers.ReLU,
                             name='Stage2_Conv_96a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=192,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Stage2_Conv1x1_192', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=192,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='Stage2_DepthwiseConv192', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=96,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.Linear,
                        name='Stage2_Conv1x1_96b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=192,
                             convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                             convInit=Layers.XavierInit, convPadding='SAME',
                             biasInit=Layers.ConstInit(0.0),
                             batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=Layers.ReLU,
                             pool=True, poolSize=[3, 3], poolStride=[2, 2],
                             poolType=Layers.MaxPool, poolPadding='SAME',
                             name='Stage3_Conv_192a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=384,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Stage3_Conv1x1_384', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=384,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='Stage3_DepthwiseConv384', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=192,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.Linear,
                        name='Stage3_Conv1x1_192b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    
    toconcat = Layers.Conv2D(concated, convChannels=384,
                             convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                             convInit=Layers.XavierInit, convPadding='SAME',
                             biasInit=Layers.ConstInit(0.0),
                             batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=Layers.ReLU,
                             pool=True, poolSize=[3, 3], poolStride=[2, 2],
                             poolType=Layers.MaxPool, poolPadding='SAME',
                             name='Stage4_Conv_384a', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=768,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Stage4_Conv1x1_768', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=768,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='Stage4_DepthwiseConv768', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=384,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.Linear,
                        name='Stage4_Conv1x1_384b', dtype=tf.float32)
    layers.append(net)
    
    concated = tf.concat([toconcat.output, net.output], axis=3)
    # again, this layer does not show up in diagram
    toadd = Layers.Conv2D(concated, convChannels=768,
                          convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          activation=Layers.Linear,
                          name='SepConv768Toadd', dtype=tf.float32)
    layers.append(toadd)
    conved = toadd.output
    
    for idx in range(numMiddle):
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768,
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                               convInit=Layers.XavierInit, convPadding='SAME',
                               biasInit=Layers.ConstInit(0.0),
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768,
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                               convInit=Layers.XavierInit, convPadding='SAME',
                               biasInit=Layers.ConstInit(0.0),
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=768,
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                               convInit=Layers.XavierInit, convPadding='SAME',
                               biasInit=Layers.ConstInit(0.0),
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved

    # this skip connection does not show up in paper
    toadd = Layers.Conv2D(conved, convChannels=1536,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          pool=True, poolSize=[3, 3], poolStride=[2, 2],
                          poolType=Layers.MaxPool, poolPadding='SAME',
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit768_1')
    layers.append(net)

    # this does not show up in diagram in paper
    toconcat = Layers.Conv2D(net.output, convChannels=768,
                             convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                             convInit=Layers.XavierInit, convPadding='SAME',
                             biasInit=Layers.ConstInit(0.0),
                             batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                             activation=Layers.ReLU,
                             pool=True, poolSize=[3, 3], poolStride=[2, 2],
                             poolType=Layers.MaxPool, poolPadding='SAME',
                             name='ConvExit768_1', dtype=tf.float32)
    layers.append(toconcat)
    
    net = Layers.Conv2D(toconcat.output, convChannels=1536,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='Exit_Conv1x1_1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.DepthwiseConv2D(net.output, convChannels=1536,
                                 convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                                 convInit=Layers.XavierInit, convPadding='SAME',
                                 biasInit=Layers.ConstInit(0.0),
                                 bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                                 activation=Layers.ReLU,
                                 name='Exit_DepthwiseConv1536', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=768,
                        convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.Linear,
                        name='Exit_Conv1x1_768b', dtype=tf.float32)
    layers.append(net)
   
    concated = tf.concat([toconcat.output, net.output], axis=3)
    added = concated + toadd.output

    # does not show up in diagram in paper
    net = Layers.SepConv2D(added, convChannels=2048,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net


def Xception(standardized, step, ifTest, layers, numMiddle=8):
    
    net = Layers.Conv2D(standardized, convChannels=32,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='ConvEntry32_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.Conv2D(net.output, convChannels=64,
                        convKernel=[3, 3], convStride=[1, 1], conv_weight_decay=wd,
                        convInit=Layers.XavierInit, convPadding='SAME',
                        biasInit=Layers.ConstInit(0.0),
                        batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                        activation=Layers.ReLU,
                        name='ConvEntry64_1', dtype=tf.float32)
    layers.append(net)
    
    toadd = Layers.Conv2D(net.output, convChannels=128,
                          convKernel=[1, 1], convStride=[1, 1], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvEntry1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='ConvEntry128_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=128,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           name='ConvEntry128_2', dtype=tf.float32)
    layers.append(net)
    
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=256,
                          convKernel=[1, 1], convStride=[2, 2], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvEntry1x1_2', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry256_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=256,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='ConvEntry256_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=256,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           pool=True, poolSize=[3, 3], poolStride=[2, 2],
                           poolType=Layers.MaxPool, poolPadding='SAME',
                           name='ConvEntry256_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    toadd = Layers.Conv2D(added, convChannels=728,
                          convKernel=[1, 1], convStride=[2, 2], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvEntry1x1_3', dtype=tf.float32)
    layers.append(toadd)
    
    acted = Layers.Activation(added, Layers.ReLU, name='ReLUEntry728_0')
    layers.append(acted)
    net = Layers.SepConv2D(acted.output, convChannels=728,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='ConvEntry728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           pool=True, poolSize=[3, 3], poolStride=[2, 2],
                           poolType=Layers.MaxPool, poolPadding='SAME',
                           name='ConvEntry728_2', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    conved = added
    
    for idx in range(numMiddle): 
        net = Layers.Activation(conved, Layers.ReLU, name='ActMiddle'+str(idx)+'_1')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728,
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                               convInit=Layers.XavierInit, convPadding='SAME',
                               biasInit=Layers.ConstInit(0.0),
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle'+str(idx)+'_1', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_2')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728,
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                               convInit=Layers.XavierInit, convPadding='SAME',
                               biasInit=Layers.ConstInit(0.0),
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle'+str(idx)+'_2', dtype=tf.float32)
        layers.append(net)
        net = Layers.Activation(net.output, Layers.ReLU, name='ReLUMiddle'+str(idx)+'_3')
        layers.append(net)
        net = Layers.SepConv2D(net.output, convChannels=728,
                               convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                               convInit=Layers.XavierInit, convPadding='SAME',
                               biasInit=Layers.ConstInit(0.0),
                               bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                               name='ConvMiddle'+str(idx)+'_3', dtype=tf.float32)
        layers.append(net)
        conved = net.output + conved
    
    toadd = Layers.Conv2D(conved, convChannels=1024,
                          convKernel=[1, 1], convStride=[2, 2], conv_weight_decay=wd,
                          convInit=Layers.XavierInit, convPadding='SAME',
                          biasInit=Layers.ConstInit(0.0),
                          batch_normalisation=True, step=step, ifTest=ifTest, epsilon=1e-5,
                          name='ConvExit1x1_1', dtype=tf.float32)
    layers.append(toadd)
    
    net = Layers.Activation(conved, Layers.ReLU, name='ActExit728_1')
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=728,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='ConvExit728_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=1024,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           pool=True, poolSize=[3, 3], poolStride=[2, 2],
                           poolType=Layers.MaxPool, poolPadding='SAME',
                           name='ConvExit1024_1', dtype=tf.float32)
    layers.append(net)
    added = toadd.output + net.output
    
    net = Layers.SepConv2D(added, convChannels=1536,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='ConvExit1536_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.SepConv2D(net.output, convChannels=2048,
                           convKernel=[3, 3], convStride=[1, 1], convWD=wd,
                           convInit=Layers.XavierInit, convPadding='SAME',
                           biasInit=Layers.ConstInit(0.0),
                           bn=True, step=step, ifTest=ifTest, epsilon=1e-5,
                           activation=Layers.ReLU,
                           name='ConvExit2048_1', dtype=tf.float32)
    layers.append(net)
    net = Layers.GlobalAvgPool(net.output, name='GlobalAvgPool')
    layers.append(net)
    
    return net
