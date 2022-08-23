import random
import h5py
import numpy as np

import tensorflow as tf

import preproc
import layers
import nets


def load_HDF5():
    with h5py.File('CIFAR10.h5', 'r') as f:
        dataTrain = np.array(f['Train']['images'])
        labelsTrain = np.array(f['Train']['labels'])
        dataTest = np.array(f['Test']['images'])
        labelsTest = np.array(f['Test']['labels'])

    return dataTrain, labelsTrain, dataTest, labelsTest


Hyper_Params = {'BatchSize': 200,
                'LearningRate': 1e-3,
                'MinLearningRate': 2 * 1e-5,
                'DecayRate': 0.9,
                'DecayAfter': 300,
                'ValidateAfter': 300,
                'TestSteps': 50,
                'TotalSteps': 30000}


class NetImageNet(nets.Net):

    def __init__(self, image_shape, architecture, hyper_params=None):
        nets.Net.__init__(self)

        if hyper_params is None:
            hyper_params = Hyper_Params

        self._init = False
        self._hyper_params = hyper_params

        with self._graph.as_default():
            # Inputs
            self._images = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None] + image_shape,
                                                    name='CIFAR10_images')
            self._labels = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None],
                                                    name='CIFAR10_labels_class')

            # define network body
            self._body = self.body(self._images, architecture)
            self._inference = self.inference(self._body)
            # defines accuracy metric. checks if inference output is equal to labels, and computes an average of the
            # number of times the output is correct
            self._accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(self._inference, self._labels), tf.float32))
            self._loss = self.loss(self._body, self._labels)

            # why reset the loss to 0 after defining it as a Tensor?
            self._loss = 0
            # add losses of all layers to NN loss tensor
            for elem in self._layers:
                if len(elem.losses) > 0:
                    for tmp in elem.losses:
                        self._loss += tmp

            self._update_ops = []
            for elem in self._layers:
                if len(elem.update_ops) > 0:
                    for tmp in elem.update_ops:
                        self._update_ops.append(tmp)

            print(self.summary)
            print("\n Begin Training: \n")

            # Saver
            self._saver = tf.compat.v1.train.Saver(max_to_keep=5)

    def preproc(self, images):
        # normalise images
        casted = tf.cast(images, tf.float32)
        standardized = tf.identity(casted / 127.5 - 1.0, name='training_standardized')

        return standardized

    def body(self, images, architecture, num_middle=2, for_generator=False):
        net_output = super().body(images, architecture, num_middle=num_middle)
        # add label for classification with 10 labels. Outputs raw logits.
        class10 = layers.FullyConnected(net_output, outputSize=10, weightInit=layers.XavierInit, l2_constant=1e-4,
                                        biasInit=layers.const_init(0.0),
                                        activation=layers.Linear,
                                        name='FC_Coarse', dtype=tf.float32)
        self._layers.append(class10)

        return class10.output

    def inference(self, logits):
        return tf.argmax(input=logits, axis=-1, name='inference')

    def loss(self, logits, labels, name='cross_entropy'):
        net = layers.CrossEntropy(logits, labels, name=name)
        self._layers.append(net)
        return net.output

    def train(self, training_data_generator, test_data_generator, path_load=None, path_save=None):
        with self._graph.as_default():
            # define decaying learning rate
            self._lr = tf.compat.v1.train.exponential_decay(self._hyper_params['LearningRate'],
                                                            global_step=self._step,
                                                            decay_steps=self._hyper_params['DecayAfter'],
                                                            decay_rate=self._hyper_params['DecayRate'])
            self._lr += self._hyper_params['MinLearningRate']

            # define optimiser
            self._optimizer = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-8).minimize(self._loss,
                                                                                                global_step=self._step)
            # Initialize all
            self._sess.run(tf.compat.v1.global_variables_initializer())
            # check if it should re-start training from a known checkpoint
            if path_load is not None:
                self.load(path_load)
            self.evaluate(test_data_generator)

            # set testing flag to false
            self._sess.run([self._phaseTrain])
            if path_save is not None:
                self.save(path_save)

            for _ in range(self._hyper_params['TotalSteps']):
                data, label = next(training_data_generator)

                # calculate loss and accuracy and perform one minimisation step
                loss, accuracy, step, _ = self._sess.run([self._loss, self._accuracy, self._step, self._optimizer],
                                                         feed_dict={self._images: data,
                                                                    self._labels: label})
                self._sess.run(self._update_ops)
                print('\rStep: ', step, '; L: %.3f' % loss, '; A: %.3f' % accuracy, end='')

                if step % self._hyper_params['ValidateAfter'] == 0:
                    self.evaluate(test_data_generator)

                    # save state of model at each evaluation step
                    if path_save is not None:
                        self.save(path_save)

                    # set test flag back to False
                    self._sess.run([self._phaseTrain])

    def evaluate(self, genTest, path=None):
        if path is not None:
            self.load(path)

        total_loss = 0.0
        total_accuracy = 0.0
        # set test flag to true
        self._sess.run([self._phaseTest])
        for _ in range(self._hyper_params['TestSteps']):
            data, label = next(genTest)
            loss, accuracy = self._sess.run([self._loss, self._accuracy],
                                            feed_dict={self._images: data,
                                                       self._labels: label})
            total_loss += loss
            total_accuracy += accuracy

        loss = total_loss / self._hyper_params['TestSteps']
        accuracy = total_accuracy / self._hyper_params['TestSteps']
        print('\nTest: Loss: ', loss, '; Accu: ', accuracy)

    def infer(self, images):
        self._sess.run([self._phaseTest])
        return self._sess.run(self._inference, feed_dict={self._images: images})


if __name__ == '__main__':
    net = NetImageNet([32, 32, 3], "SmallNet")
    batchTrain, batchTest = get_data_generators(batch_size=Hyper_Params['BatchSize'], image_size=[32, 32, 3])
    net.train(batchTrain, batchTest, path_save='./ClassifyCIFAR10/netcifar10.ckpt')
