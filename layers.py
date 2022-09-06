import functools
import tensorflow as tf

from tensorflow.python.training.moving_averages import assign_moving_average

weight_l2_loss = 1e-4

# Initializers
XavierInit = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
Norm01Init = tf.compat.v1.truncated_normal_initializer(0.0, stddev=0.1)


def normal_init(stddev, dtype=tf.float32):
    return tf.compat.v1.truncated_normal_initializer(0.0, stddev=stddev, dtype=dtype)


def const_init(const, dtype=tf.float32):
    return tf.compat.v1.constant_initializer(const, dtype=dtype)


# Activations
Linear = tf.identity
Sigmoid = tf.nn.sigmoid
Tanh = tf.nn.tanh
ReLU = tf.nn.relu
ELU = tf.nn.elu
Softmax = tf.nn.softmax


def LeakyReLU(alpha=0.2):
    return functools.partial(tf.nn.leaky_relu, alpha=alpha)


# Poolings
AvgPool = tf.nn.avg_pool2d
MaxPool = tf.nn.max_pool2d

l2_regulariser = tf.keras.regularizers.L2(l2=1e-4 * 0.5)


def conv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True,
                               activation=activation, kernel_regularizer=l2_regulariser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def depthwise_conv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    channels_in = x.shape[3]
    assert filters % channels_in == 0
    depth_multiplier = int(filters / channels_in)

    x = tf.keras.layers.DepthwiseConv2D(depth_multiplier=depth_multiplier, kernel_size=kernel_size, strides=strides,
                                        padding='same', use_bias=True, activation=activation,
                                        depthwise_regularizer=l2_regulariser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def sep_conv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    x = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                        use_bias=True, activation=activation, depthwise_regularizer=l2_regulariser,
                                        pointwise_regularizer=l2_regulariser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def deconv2d_bn(x, filters, kernel_size, strides=1, activation=None):
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                        use_bias=True, activation=activation, kernel_regularizer=l2_regulariser)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


class Layer(object):

    def __init__(self):
        self._output = None
        self._variables = []
        self._updateOps = []
        self._losses = []

    @property
    def type(self):
        return 'Layer'

    @property
    def output(self):
        return self._output

    @property
    def variables(self):
        return self._variables

    @property
    def update_ops(self):
        return self._updateOps

    @property
    def losses(self):
        return self._losses

    @property
    def summary(self):
        return 'Layer: the parent class of all layers'


# Convolution


class Conv2D(Layer):

    def __init__(self, feature, convChannels,
                 convKernel=[3, 3], convStride=[1, 1], l2_constant=weight_l2_loss, convInit=XavierInit,
                 convPadding='SAME', bias=True, biasInit=const_init(0.0),
                 batch_norm=False, step=None, ifTest=None, epsilon=1e-5,
                 activation=Linear, pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool,
                 poolPadding='SAME', reuse=False, name=None, dtype=tf.float32):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._losses = []
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._sizeKernel = convKernel + [feature.get_shape().as_list()[3], convChannels]
            self._strideConv = [1] + convStride + [1]
            self._typeConvPadding = convPadding
            self._weights = tf.compat.v1.get_variable(scope.name + '_weights',
                                                      self._sizeKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.conv2d(input=feature, filters=self._weights, strides=self._strideConv,
                                padding=self._typeConvPadding,
                                name=scope.name + '_conv2d')
            self._variables.append(self._weights)
            if l2_constant is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), l2_constant, name=scope.name + 'l2_wd')
                self._losses.append(decay)

            # tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias:
                self._bias = tf.compat.v1.get_variable(scope.name + '_bias', [convChannels],
                                                       initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)

            self._bn = batch_norm
            if batch_norm:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams = [conv.shape[-1]]
                self._offset = tf.compat.v1.get_variable(scope.name + '_offset',
                                                         shapeParams, initializer=const_init(0.0), dtype=dtype)
                self._scale = tf.compat.v1.get_variable(scope.name + '_scale',
                                                        shapeParams, initializer=const_init(1.0), dtype=dtype)
                self._movMean = tf.compat.v1.get_variable(scope.name + '_movMean', shapeParams, trainable=False,
                                                          initializer=const_init(0.0), dtype=dtype, use_resource=True)
                self._movVar = tf.compat.v1.get_variable(scope.name + '_movVar', shapeParams, trainable=False,
                                                         initializer=const_init(1.0), dtype=dtype, use_resource=True)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon = epsilon

                def trainMeanVar():
                    mean, var = tf.nn.moments(x=conv, axes=list(range(len(conv.shape) - 1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9),
                                                  assign_moving_average(self._movVar, var, 0.9)]):
                        self._trainMean = tf.identity(mean)
                        self._trainVar = tf.identity(var)
                    return self._trainMean, self._trainVar

                self._actualMean, self._actualVar = tf.cond(pred=ifTest, true_fn=lambda: (self._movMean, self._movVar),
                                                            false_fn=trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar,
                                                 self._offset, self._scale, self._epsilon,
                                                 name=scope.name + '_batch_normalization')

            self._activation = activation
            if activation is not None:
                activated = activation(conv, name=scope.name + '_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling = [1] + poolSize + [1]
                self._stridePooling = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling,
                                  padding=self._typePoolPadding,
                                  name=scope.name + '_pooling')
            else:
                self._sizePooling = [0]
                self._stridePooling = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled

    @property
    def type(self):
        return 'Conv2D'

    @property
    def summary(self):
        if isinstance(self._activation, functools.partial):
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Kernel Size: ' + str(self._sizeKernel) + '; ' +
                'Conv Stride: ' + str(self._strideConv) + '; ' + 'Conv Padding: ' + self._typeConvPadding + '; ' +
                'Batch Normalization: ' + str(self._bn) + '; ' +
                'Pooling Size: ' + str(self._sizePooling) + '; ' + 'Pooling Size: ' + str(self._stridePooling) + '; ' +
                'Pooling Padding: ' + self._typePoolPadding + '; ' + 'Activation: ' + activation + ']')


class DeConv2D(Layer):
    def __init__(self, feature, convChannels, shapeOutput=None,
                 convKernel=[3, 3], convStride=[1, 1], l2_constant=weight_l2_loss, convInit=XavierInit, convPadding='SAME',
                 bias=True, biasInit=const_init(0.0), batch_norm=False, step=None, ifTest=None, epsilon=1e-5,
                 activation=Linear, pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool,
                 poolPadding='SAME', reuse=False, name=None, dtype=tf.float32):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._losses = []
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._sizeKernel = convKernel + [convChannels, feature.get_shape().as_list()[3]]
            self._strideConv = [1] + convStride + [1]
            if shapeOutput is None:
                self._shapeOutput = tf.TensorShape(
                    [feature.get_shape().as_list()[0], feature.get_shape().as_list()[1] * convStride[0],
                     feature.get_shape().as_list()[2] * convStride[1], convChannels])
            else:
                self._shapeOutput = tf.TensorShape([feature.shape[0]] + shapeOutput + [convChannels])
            self._typeConvPadding = convPadding
            self._weights = tf.compat.v1.get_variable(scope.name + '_weights',
                                                      self._sizeKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.conv2d_transpose(feature, self._weights, self._shapeOutput, self._strideConv,
                                          padding=self._typeConvPadding,
                                          name=scope.name + '_conv2d_transpose')
            self._variables.append(self._weights)
            if l2_constant is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), l2_constant, name=scope.name + 'l2_wd')
                self._losses.append(decay)

            # tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias:
                self._bias = tf.compat.v1.get_variable(scope.name + '_bias', [convChannels],
                                                       initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)

            self._bn = batch_norm
            if batch_norm:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams = [conv.shape[-1]]
                self._offset = tf.compat.v1.get_variable(scope.name + '_offset',
                                                         shapeParams, initializer=const_init(0.0), dtype=dtype)
                self._scale = tf.compat.v1.get_variable(scope.name + '_scale',
                                                        shapeParams, initializer=const_init(1.0), dtype=dtype)
                self._movMean = tf.compat.v1.get_variable(scope.name + '_movMean', shapeParams, trainable=False,
                                                          initializer=const_init(0.0), dtype=dtype, use_resource=True)
                self._movVar = tf.compat.v1.get_variable(scope.name + '_movVar', shapeParams, trainable=False,
                                                         initializer=const_init(1.0), dtype=dtype, use_resource=True)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon = epsilon

                def trainMeanVar():
                    mean, var = tf.nn.moments(x=conv, axes=list(range(len(conv.shape) - 1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9),
                                                  assign_moving_average(self._movVar, var, 0.9)]):
                        self._trainMean = tf.identity(mean)
                        self._trainVar = tf.identity(var)
                    return self._trainMean, self._trainVar

                self._actualMean, self._actualVar = tf.cond(pred=ifTest, true_fn=lambda: (self._movMean, self._movVar),
                                                            false_fn=trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar,
                                                 self._offset, self._scale, self._epsilon,
                                                 name=scope.name + '_batch_normalization')

            self._activation = activation
            if activation is not None:
                activated = activation(conv, name=scope.name + '_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling = [1] + poolSize + [1]
                self._stridePooling = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling,
                                  padding=self._typePoolPadding,
                                  name=scope.name + '_pooling')
            else:
                self._sizePooling = [0]
                self._stridePooling = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled

    @property
    def type(self):
        return 'DeConv2D'

    @property
    def summary(self):
        if isinstance(self._activation, functools.partial):
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Kernel Size: ' + str(self._sizeKernel) + '; ' +
                'Conv Stride: ' + str(self._strideConv) + '; ' + 'Conv Padding: ' + self._typeConvPadding + '; ' +
                'Batch Normalization: ' + str(self._bn) + '; ' +
                'Pooling Size: ' + str(self._sizePooling) + '; ' + 'Pooling Size: ' + str(self._stridePooling) + '; ' +
                'Pooling Padding: ' + self._typePoolPadding + '; ' + 'Activation: ' + activation + ']')


class SepConv2D(Layer):

    def __init__(self, feature, convChannels,
                 convKernel=[3, 3], convStride=[1, 1], l2_constant=weight_l2_loss, convInit=XavierInit, convPadding='SAME',
                 bias=True, biasInit=const_init(0.0), batch_norm=False, step=None, ifTest=None, epsilon=1e-5,
                 activation=Linear, pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool,
                 poolPadding='SAME', reuse=False, name=None, dtype=tf.float32):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._losses = []
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._sizeDepthKernel = convKernel + [feature.get_shape().as_list()[3], 1]
            self._sizePointKernel = [1, 1] + [feature.get_shape().as_list()[3], convChannels]
            self._strideConv = [1] + convStride + [1]
            self._typeConvPadding = convPadding
            self._weightsDepth = tf.compat.v1.get_variable(scope.name + '_weightsDepth',
                                                           self._sizeDepthKernel, initializer=convInit, dtype=dtype)
            self._weightsPoint = tf.compat.v1.get_variable(scope.name + '_weightsPoint',
                                                           self._sizePointKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.separable_conv2d(input=feature, depthwise_filter=self._weightsDepth,
                                          pointwise_filter=self._weightsPoint,
                                          strides=self._strideConv, padding=self._typeConvPadding,
                                          name=scope.name + '_sep_conv')
            self._variables.append(self._weightsDepth)
            self._variables.append(self._weightsPoint)
            if l2_constant is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weightsDepth), l2_constant, name=scope.name + 'l2_wd_depth')
                self._losses.append(decay)
                decay = tf.multiply(tf.nn.l2_loss(self._weightsPoint), l2_constant, name=scope.name + 'l2_wd_point')
                self._losses.append(decay)

            # tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias:
                self._bias = tf.compat.v1.get_variable(scope.name + '_bias', [convChannels],
                                                       initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)

            self._bn = batch_norm
            if batch_norm:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams = [conv.shape[-1]]
                self._offset = tf.compat.v1.get_variable(scope.name + '_offset',
                                                         shapeParams, initializer=const_init(0.0), dtype=dtype)
                self._scale = tf.compat.v1.get_variable(scope.name + '_scale',
                                                        shapeParams, initializer=const_init(1.0), dtype=dtype)
                self._movMean = tf.compat.v1.get_variable(scope.name + '_movMean', shapeParams, trainable=False,
                                                          initializer=const_init(0.0), dtype=dtype, use_resource=True)
                self._movVar = tf.compat.v1.get_variable(scope.name + '_movVar', shapeParams, trainable=False,
                                                         initializer=const_init(1.0), dtype=dtype, use_resource=True)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon = epsilon

                def trainMeanVar():
                    mean, var = tf.nn.moments(x=conv, axes=list(range(len(conv.shape) - 1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9),
                                                  assign_moving_average(self._movVar, var, 0.9)]):
                        self._trainMean = tf.identity(mean)
                        self._trainVar = tf.identity(var)
                    return self._trainMean, self._trainVar

                self._actualMean, self._actualVar = tf.cond(pred=ifTest, true_fn=lambda: (self._movMean, self._movVar),
                                                            false_fn=trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar,
                                                 self._offset, self._scale, self._epsilon,
                                                 name=scope.name + '_batch_normalization')

            self._activation = activation
            if activation is not None:
                activated = activation(conv, name=scope.name + '_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling = [1] + poolSize + [1]
                self._stridePooling = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling,
                                  padding=self._typePoolPadding,
                                  name=scope.name + '_pooling')
            else:
                self._sizePooling = [0]
                self._stridePooling = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled

    @property
    def type(self):
        return 'SepConv2D'

    @property
    def summary(self):
        if isinstance(self._activation, functools.partial):
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Kernel Size: ' + str(self._sizeDepthKernel) + ', ' + str(self._sizePointKernel) +
                'Conv Stride: ' + str(self._strideConv) + '; ' + 'Conv Padding: ' + self._typeConvPadding + '; ' +
                'Batch Normalization: ' + str(self._bn) + '; ' +
                'Pooling Size: ' + str(self._sizePooling) + '; ' + 'Pooling Size: ' + str(self._stridePooling) + '; ' +
                'Pooling Padding: ' + self._typePoolPadding + '; ' + 'Activation: ' + activation + ']')


class DepthwiseConv2D(Layer):

    def __init__(self, feature, convChannels,
                 convKernel=[3, 3], convStride=[1, 1], l2_constant=weight_l2_loss, convInit=XavierInit, convPadding='SAME',
                 bias=True, biasInit=const_init(0.0), batch_norm=False, step=None, ifTest=None, epsilon=1e-5,
                 activation=Linear, pool=False, poolSize=[3, 3], poolStride=[2, 2], poolType=MaxPool,
                 poolPadding='SAME', reuse=False, name=None, dtype=tf.float32):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._losses = []
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._sizeDepthKernel = convKernel + [feature.get_shape().as_list()[3],
                                                  int(convChannels / feature.get_shape().as_list()[3])]
            self._strideConv = [1] + convStride + [1]
            self._typeConvPadding = convPadding
            self._weightsDepth = tf.compat.v1.get_variable(scope.name + '_weightsDepth',
                                                           self._sizeDepthKernel, initializer=convInit, dtype=dtype)
            conv = tf.nn.depthwise_conv2d(input=feature, filter=self._weightsDepth, strides=self._strideConv,
                                          padding=self._typeConvPadding,
                                          name=scope.name + '_depthwise_conv')
            self._variables.append(self._weightsDepth)
            if l2_constant is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weightsDepth), l2_constant, name=scope.name + 'l2_wd_depth')
                self._losses.append(decay)

            # tf.nn.conv2d(pooling, kernel, [1, 1, 1, 1], padding=Padding)
            if bias:
                self._bias = tf.compat.v1.get_variable(scope.name + '_bias', [convChannels],
                                                       initializer=biasInit, dtype=dtype)
                conv = conv + self._bias
                self._variables.append(self._bias)

            self._bn = batch_norm
            if batch_norm:
                assert (step is not None), "step parameter must not be None. "
                assert (ifTest is not None), "ifTest parameter must not be None. "
                shapeParams = [conv.shape[-1]]
                self._offset = tf.compat.v1.get_variable(scope.name + '_offset',
                                                         shapeParams, initializer=const_init(0.0), dtype=dtype)
                self._scale = tf.compat.v1.get_variable(scope.name + '_scale',
                                                        shapeParams, initializer=const_init(1.0), dtype=dtype)
                self._movMean = tf.compat.v1.get_variable(scope.name + '_movMean', shapeParams, trainable=False,
                                                          initializer=const_init(0.0), dtype=dtype, use_resource=True)
                self._movVar = tf.compat.v1.get_variable(scope.name + '_movVar', shapeParams, trainable=False,
                                                         initializer=const_init(1.0), dtype=dtype, use_resource=True)
                self._variables.append(self._scale)
                self._variables.append(self._offset)
                self._epsilon = epsilon

                def trainMeanVar():
                    mean, var = tf.nn.moments(x=conv, axes=list(range(len(conv.shape) - 1)))
                    with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9),
                                                  assign_moving_average(self._movVar, var, 0.9)]):
                        self._trainMean = tf.identity(mean)
                        self._trainVar = tf.identity(var)
                    return self._trainMean, self._trainVar

                self._actualMean, self._actualVar = tf.cond(pred=ifTest, true_fn=lambda: (self._movMean, self._movVar),
                                                            false_fn=trainMeanVar)
                conv = tf.nn.batch_normalization(conv, self._actualMean, self._actualVar,
                                                 self._offset, self._scale, self._epsilon,
                                                 name=scope.name + '_batch_normalization')

            self._activation = activation
            if activation is not None:
                activated = activation(conv, name=scope.name + '_activation')
            else:
                activated = conv
            if pool:
                self._sizePooling = [1] + poolSize + [1]
                self._stridePooling = [1] + poolStride + [1]
                self._typePoolPadding = poolPadding
                pooled = poolType(activated, ksize=self._sizePooling, strides=self._stridePooling,
                                  padding=self._typePoolPadding,
                                  name=scope.name + '_pooling')
            else:
                self._sizePooling = [0]
                self._stridePooling = [0]
                self._typePoolPadding = 'NONE'
                pooled = activated
        self._output = pooled

    @property
    def type(self):
        return 'DepthwiseConv2D'

    @property
    def summary(self):
        if isinstance(self._activation, functools.partial):
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Kernel Size: ' + str(self._sizeDepthKernel) + ',; ' +
                'Conv Stride: ' + str(self._strideConv) + '; ' + 'Conv Padding: ' + self._typeConvPadding + '; ' +
                'Batch Normalization: ' + str(self._bn) + '; ' +
                'Pooling Size: ' + str(self._sizePooling) + '; ' + 'Pooling Size: ' + str(self._stridePooling) + '; ' +
                'Pooling Padding: ' + self._typePoolPadding + '; ' + 'Activation: ' + activation + ']')


# Normalizations

class LocalResponseNorm(Layer):

    def __init__(self, feature, depth_radius=5, bias=1, alpha=1, beta=0.5, reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._depthRadius = depth_radius
        self._bias = bias
        self._alpha = alpha
        self._beta = beta
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._output = tf.nn.lrn(feature, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta,
                                     name=scope.name)

    @property
    def type(self):
        return 'LRNorm'

    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + 'Output Size: ' + str(self._output.shape) + '; ' +
                '; ' + 'Depth Radius: ' + self._depthRadius + '; ' +
                'Bias: ' + self._bias + '; ' + 'Alpha: ' + self._alpha + '; ' + 'Beta: ' + self._beta + ']')


class BatchNorm(Layer):

    def __init__(self, feature, step, ifTest, epsilon=1e-5, reuse=False, name=None, dtype=tf.float32):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            shapeParams = [feature.shape[-1]]
            self._offset = tf.compat.v1.get_variable(scope.name + '_offset',
                                                     shapeParams, initializer=const_init(0.0), dtype=dtype)
            self._scale = tf.compat.v1.get_variable(scope.name + '_scale',
                                                    shapeParams, initializer=const_init(1.0), dtype=dtype)
            self._movMean = tf.compat.v1.get_variable(scope.name + '_movMean', shapeParams, trainable=False,
                                                      initializer=const_init(0.0), dtype=dtype, use_resource=True)
            self._movVar = tf.compat.v1.get_variable(scope.name + '_movVar', shapeParams, trainable=False,
                                                     initializer=const_init(1.0), dtype=dtype, use_resource=True)
            self._variables.append(self._scale)
            self._variables.append(self._offset)
            self._epsilon = epsilon

            def trainMeanVar():
                mean, var = tf.nn.moments(x=feature, axes=list(range(len(feature.shape) - 1)))
                with tf.control_dependencies([assign_moving_average(self._movMean, mean, 0.9),
                                              assign_moving_average(self._movVar, var, 0.9)]):
                    self._trainMean = tf.identity(mean)
                    self._trainVar = tf.identity(var)
                return self._trainMean, self._trainVar

            self._actualMean, self._actualVar = tf.cond(pred=ifTest, true_fn=lambda: (self._movMean, self._movVar),
                                                        false_fn=trainMeanVar)
            self._output = tf.nn.batch_normalization(feature, self._actualMean, self._actualVar,
                                                     self._offset, self._scale, self._epsilon,
                                                     name=scope.name + '_batch_normalization')

    @property
    def type(self):
        return 'BatchNorm'

    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Epsilon: ' + str(self._epsilon) + ']')


class Dropout(Layer):

    def __init__(self, feature, ifTest, rateKeep=0.5, reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._rateKeep = rateKeep
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._keepProb = tf.Variable(rateKeep, trainable=False)

            def phaseTest():
                return tf.compat.v1.assign(self._keepProb, 1.0)

            def phaseTrain():
                return tf.compat.v1.assign(self._keepProb, rateKeep)

            with tf.control_dependencies([tf.cond(pred=ifTest, true_fn=phaseTest, false_fn=phaseTrain)]):
                self._output = tf.nn.dropout(feature, 1 - (self._keepProb), name=scope.name + '_dropout')

    @property
    def type(self):
        return 'Dropout'

    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Keep Rate: ' + str(self._rateKeep) + ']')


class Flatten(Layer):
    def __init__(self, feature, name="Flatten"):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self);
        self._name = name
        size = feature.shape[1]
        for elem in feature.shape[2:]:
            size *= elem
        self._output = tf.reshape(feature, [-1, size])

    @property
    def type(self):
        return 'Flatten'

    @property
    def summary(self):
        return self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + ']'


class FullyConnected(Layer):

    def __init__(self, feature, outputSize, weightInit=XavierInit, l2_constant=weight_l2_loss,
                 bias=True, biasInit=const_init(0.0),
                 activation=ReLU,
                 reuse=False, name=None, dtype=tf.float32):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._sizeWeights = [feature.get_shape().as_list()[1], outputSize]
            self._weights = tf.compat.v1.get_variable(scope.name + '_weights',
                                                      self._sizeWeights, initializer=weightInit, dtype=dtype)
            self._variables.append(self._weights)
            if l2_constant is not None:
                decay = tf.multiply(tf.nn.l2_loss(self._weights), l2_constant, name=scope.name + 'l2_wd')
                self._losses.append(decay)
            if bias:
                self._bias = tf.compat.v1.get_variable(scope.name + '_bias', [outputSize],
                                                       initializer=biasInit, dtype=dtype)
                self._variables.append(self._bias)
            else:
                self._bias = tf.constant(0.0, dtype=dtype)

            self._output = tf.add(tf.matmul(feature, self._weights), self._bias, name=scope.name + '_fully_connected')
            self._activation = activation
            if activation is not None:
                self._output = activation(self._output, name=scope.name + '_activation')

    @property
    def type(self):
        return 'FullyConnected'

    @property
    def summary(self):
        if isinstance(self._activation, functools.partial):
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Weight Size: ' + str(self._weights.shape) + '; ' +
                'Bias Size: ' + str(self._bias.shape) + '; ' +
                'Activation: ' + activation + ']')


class Activation(Layer):

    def __init__(self, feature, activation=Linear, reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._activation = activation
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._output = activation(feature, name=scope.name + '_activation')

    @property
    def type(self):
        return 'Activation'

    @property
    def summary(self):
        if isinstance(self._activation, functools.partial):
            activation = self._activation.func.__name__
        else:
            activation = self._activation.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Activation: ' + activation + ']')


class Pooling(Layer):

    def __init__(self, feature, pool=MaxPool, size=[2, 2], stride=[2, 2], padding='SAME', reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        self._typePool = pool
        self._sizePooling = [1] + size + [1]
        self._stridePooling = [1] + stride + [1]
        self._typePoolPadding = padding
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._output = self._typePool(feature, ksize=self._sizePooling, strides=self._stridePooling,
                                          padding=self._typePoolPadding,
                                          name=scope.name + '_pooling')

    @property
    def type(self):
        return 'Pooling'

    @property
    def summary(self):
        if isinstance(self._typePool, functools.partial):
            pooltype = self._typePool.func.__name__
        else:
            pooltype = self._typePool.__name__
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Type: ' + pooltype + ']')


class GlobalAvgPool(Layer):

    def __init__(self, feature, reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._output = tf.reduce_mean(input_tensor=feature, axis=[1, 2], name=scope.name + '_global_avg_pool')

    @property
    def type(self):
        return 'GlobalAvgPooling'

    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Type: Global Average Pooling' + ']')


class CrossEntropy(Layer):

    def __init__(self, feature, labels, reuse=False, name=None):
        assert isinstance(feature, tf.Tensor), 'feature must be a tf.Tensor, use Layer.output to get it'

        Layer.__init__(self)
        self._name = name
        with tf.compat.v1.variable_scope(self._name, reuse=reuse) as scope:
            self._output = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=feature,
                                                                          name=scope.name + '_cross_entropy')
            self._output = tf.reduce_mean(input_tensor=self._output)
            self._losses.append(self._output)

    @property
    def type(self):
        return 'CrossEntropy'

    @property
    def summary(self):
        return (self.type + ': [Name: ' + self._name + '; ' + 'Output Size: ' + str(self._output.shape) + '; ' +
                'Activation: CrossEntropy' + ']')
