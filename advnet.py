import matplotlib.pyplot as plt
import tensorflow as tf
# gpu = tf.config.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_logical_device_configuration(
#     gpu,
#     [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])

import numpy as np

import data
import layers
import nets
import encoders
import differentiable_rendering as diff_rendering
import preproc

NoiseRange = 10.0
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits
relu = tf.keras.activations.relu


# input to generator must be textures with values normalised to -1 and 1
def create_generator(num_experts):
    textures = tf.keras.layers.Input(shape=(2048, 2048, 3), dtype=tf.float32)
    targets = tf.keras.layers.Input(shape=(1,), dtype=tf.int64)
    x = encoders.simpleNet_encoder(textures)

    # add decoder part
    x = layers.deconv2d_bn(x, filters=128, kernel_size=3, strides=2, activation=relu)
    subnets = []
    for idx in range(num_experts):
        subnet = layers.deconv2d_bn(x, filters=64, kernel_size=3, strides=2, activation=relu)
        subnet = layers.deconv2d_bn(subnet, filters=32, kernel_size=3, strides=2, activation=relu)
        subnet = layers.deconv2d_bn(subnet, filters=16, kernel_size=3, strides=2, activation=relu)
        subnet = layers.deconv2d_bn(subnet, filters=8, kernel_size=3, strides=2, activation=relu)
        subnet = layers.deconv2d_bn(subnet, filters=3, kernel_size=3, strides=2, activation=relu)
        subnets.append(tf.expand_dims(subnet, axis=-1))

    subnets = tf.concat(subnets, axis=-1)
    weights = tf.keras.layers.Dense(units=num_experts, use_bias=True,
                                    kernel_regularizer=None,
                                    activation=tf.keras.activations.softmax)(tf.one_hot(targets, 1000))

    moe = tf.transpose(a=tf.transpose(a=subnets, perm=[1, 2, 3, 0, 4]) * weights, perm=[3, 0, 1, 2, 4])
    noises = (tf.nn.tanh(tf.reduce_sum(input_tensor=moe, axis=-1)) - 0.5) * NoiseRange * 2
    print('Shape of Noises: ', noises.shape)

    return tf.keras.Model(inputs=[textures, targets], outputs=noises)


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
                         'WarmupSteps': 500,
                         'TotalSteps': 30000}


class AdvNet(nets.Net):

    def __init__(self, image_shape, architecture, hyper_params=None):
        nets.Net.__init__(self)

        if hyper_params is None:
            hyper_params = hyper_params_imagenet
        self._hyper_params = hyper_params
        self.image_shape = image_shape

        self.enemy = tf.keras.applications.xception.Xception(
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

    def create_simulator(self, architecture, num_middle=2):
        # define body of simulator
        images, output = super().create_simulator(architecture, num_middle)
        logits = tf.keras.layers.Dense(units=1000, activation=None)(output)

        return tf.keras.Model(inputs=images, outputs=logits)

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

            print('\rSimulator => Step: ', i - self._hyper_params['WarmupSteps'], end='')
            self.simulator_training_step(textures, uv_maps)

        # evaluate warmed up simulator on test data
        warmup_accuracy = 0.0
        print("Evaluating warmed up simulator:")
        for i in range(50):
            textures, uv_maps, _, _ = next(data_generator)
            warmup_accuracy += self.warm_up_evaluation(textures, uv_maps)

        warmup_accuracy = warmup_accuracy / 50
        print('\nAverage Warmup Accuracy: ', warmup_accuracy)

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

    def simulator_training_step(self, textures, uv_maps):
        # create rendering params and then render render image. We do not need to differentiate through the rendering
        # for the simulator, therefore this can be done outside of the gradient tape.
        print_error_params = diff_rendering.get_print_error_args()
        photo_error_params = diff_rendering.get_photo_error_args([self._hyper_params['BatchSize']] +
                                                                 self.image_shape + [3])
        background_colours = diff_rendering.get_background_colours()

        images = diff_rendering.render(textures, uv_maps, print_error_params, photo_error_params,
                                       background_colours)

        with tf.GradientTape() as simulator_tape:
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

    # images must have pixel values between 0 and 1
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


if __name__ == '__main__':
    net = AdvNet([299, 299], "SimpleNet")
    data_generator = data.get_adversarial_data_generators(batch_size=hyper_params_imagenet['BatchSize'])

    net.train(data_generator)
    net.plot_training_history("Adversarial CIFAR10", net._hyper_params['ValidateAfter'])
