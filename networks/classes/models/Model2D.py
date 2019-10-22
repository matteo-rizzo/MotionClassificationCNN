import tensorflow as tf
from tensorflow.python.keras import layers

from networks.classes.models.Model import Model
from networks.classes.others import Params

from inspect import signature


class SiameseConv2D(tf.keras.Model):
    def __init__(self, params, **kwargs):
        super(SiameseConv2D, self).__init__(**kwargs)
        self.img_size = params.image_size
        in_channels = params.in_channels
        out_channels = params.out_channels
        num_labels = params.num_labels
        bn_momentum = params.bn_momentum

        self.conv0 = layers.Conv2D(filters=32, kernel_size=(13, 13), padding='same', activation='relu',
                                   input_shape=(self.img_size, self.img_size, in_channels, 1), name="pippo")
        # self.bn0 = layers.BatchNormalization(momentum=bn_momentum)
        self.maxpool0 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        # self.maxpool0 = layers.Conv2D(filters=16, kernel_size=(3, 3, 2), strides=(2, 2, 1), padding='valid')

        self.conv1 = layers.Conv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')
        # self.bn1 = layers.BatchNormalization(momentum=bn_momentum)
        self.maxpool1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        # self.maxpool1 = layers.Conv2D(filters=16, kernel_size=(3, 3, 2), strides=(2, 2, 1), padding='valid')

        self.conv2 = layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')
        # self.bn2 = layers.BatchNormalization(momentum=bn_momentum)
        # self.maxpool2 = layers.MaxPool2D(pool_size=(3, 3, 2), strides=(2, 2, 1), padding='valid')

        self.conv3 = layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')
        # self.bn3 = layers.BatchNormalization(momentum=bn_momentum)
        self.maxpool3 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        # self.maxpool3 = layers.Conv2D(filters=16, kernel_size=(3, 3, 2), strides=(2, 2, 1), padding='valid')

        # self.conv4 = layers.Conv2D(filters=48, kernel_size=(3, 3, 2), padding='same', activation='relu')
        # self.bn4 = layers.BatchNormalization(momentum=bn_momentum)
        # self.maxpool4 = layers.MaxPool2D(pool_size=(3, 3, 2), strides=(2, 2, 1), padding='valid')

        # self.conv5 = layers.Conv2D(filters=80, kernel_size=(3, 3, 3), padding='same', activation='relu')
        # self.bn5 = layers.BatchNormalization(momentum=bn_momentum)
        # self.maxpool5 = layers.MaxPool2D(pool_size=(3, 3, 2), strides=(1, 1, 1), padding='valid')

        self.flatten = layers.Flatten()
        self.fc_siamese = layers.Dense(80)
        # self.bn6 = layers.BatchNormalization(momentum=bn_momentum)
        # self.dropout_siamese = layers.Dropout(params.dropout_rate)

        # output = (None, 30, 30, 128) padding=valid means resize W and H to (W-ks+1)

        # ADD DENSE LAYER ON TOP
        # To complete our model, we will feed the last output tensor from the convolutional base (of shape (X, X, 64))
        # into one or more Dense layers to perform classification.
        # Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor.
        # First, we will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top.
        # Since we have 2 output classes, we use a final Dense layer with 2 outputs and a sigmoid activation.

        self.fc0 = layers.Dense(units=160, name='dense_last', activation='relu')
        # self.bn_last = layers.BatchNormalization(momentum=bn_momentum, name='batch_norm_last')
        # self.dropout_last = layers.Dropout(rate=params.dropout_rate)
        self.classifier = layers.Dense(units=num_labels, name='classifier', activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        siamese_block = [
            self.conv0,
            # self.bn0,
            self.maxpool0,
            self.conv1,
            # self.bn1,
            self.maxpool1,
            self.conv2,
            # self.bn2,
            # self.maxpool2,
            self.conv3,
            # self.bn3,
            self.maxpool3,
            # self.conv4,
            # self.bn4,
            # self.maxpool4,
            # self.conv5,
            # self.bn5,
            # self.maxpool5,
            self.flatten,
            self.fc_siamese,
            # self.bn6,
            # self.dropout_siamese
        ]
        x = tf.unstack(inputs, axis=-1)
        siameses = []
        for i in range(2):
            siameses.append(self.call_siamese(siamese_block, x[i], training))
        block = tf.concat(siameses, axis=1)
        x = self.fc0(block)
        # x = self.bn_last(x, training=training)
        # x = self.dropout_last(x, training=training)
        out = self.classifier(x)
        return out

    @staticmethod
    def call_siamese(siamese_block, inputs, training):
        x = siamese_block[0](inputs)  # compute first output
        for layer in siamese_block[1:]:
            # invoke every layer with the output. We need to check the signature of every layer: dropout and batch_norm
            # accept also the 'training' parameter, which deactivate the layer in case of validation or test
            if 'training' not in str(signature(layer.call)):
                x = layer(x)
            else:
                x = layer(x, training=training)
        return x

    def summary(self, line_length=None, position=None, print_fn=None):
        x = tf.keras.Input(shape=(self.img_size, self.img_size, 1, 2))
        tf.keras.Model(inputs=x, outputs=self.call(x, training=True)).summary(line_length, position, print_fn)

class Model2DSiamese(Model):
    def __init__(self,
                 params: any,
                 training_set: tf.data.Dataset = None,
                 validation_set: tf.data.Dataset = None,
                 test_set: tf.data.Dataset = None) -> None:
        super().__init__(params, training_set, validation_set, test_set)
        self._build(params)

    def _build(self, params) -> None:
        """
        Builds the convolutional network.
        use_bias=False in FC/CONV nets because of this article:
        https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/
        """
        # DEFINE THE CONVOLUTIONAL BASE
        # A common pattern is used: a stack of Conv2D and MaxPooling2D layers.
        # As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size.
        # We do this by passing the argument input_shape to our first layer.
        self._model = SiameseConv2D(params)

class Model2D(Model):
    def __init__(self,
                 params: Params,
                 training_set: tf.data.Dataset = None,
                 validation_set: tf.data.Dataset = None,
                 test_set: tf.data.Dataset = None):
        super().__init__(params, training_set, validation_set, test_set)
        self._build(params)

    def _build(self, params) -> None:
        """
        Builds the convolutional network.
        """

        # Get the main parameters of the network
        img_size = params.image_size
        in_channels = params.in_channels
        out_channels = params.out_channels
        num_labels = params.num_labels
        bn_momentum = params.bn_momentum
        channels = [out_channels, out_channels * 2, out_channels * 4]

        for i, c in enumerate(channels):
            self._model.add(layers.Conv2D(filters=c,
                                          kernel_size=(7, 7),
                                          input_shape=(img_size, img_size, in_channels),
                                          data_format='channels_last',
                                          padding='same',
                                          name='conv1_{i}'.format(i=i)))
            self._model.add(layers.BatchNormalization(momentum=bn_momentum, name='batch_norm_{i}'.format(i=i)))
            self._model.add(layers.Activation('selu', name='selu1_{i}'.format(i=i)))
            self._model.add(layers.MaxPool2D(pool_size=(3, 3),
                                             strides=(3, 3),
                                             name='max_pool_{i}'.format(i=i), padding='valid'))

        self._model.add(layers.Flatten(name='flatt_1'))
        self._model.add(layers.Dense(units=128, name='last_linear', activation='linear'))

        self._model.add(layers.BatchNormalization(momentum=bn_momentum, name='batch_norm_last'))
        self._model.add(layers.Activation('selu', name='last_selu'))
        self._model.add(layers.Dense(units=num_labels, name='classifier', activation='sigmoid'))
