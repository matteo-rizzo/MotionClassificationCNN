import glob
import logging
import os
from typing import List

import tensorflow as tf
from tensorflow.python.keras import models, optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from tensorflow.python.keras.utils import plot_model

from networks.classes.others import Params


class Model:
    def __init__(self,
                 model_type: str,
                 params: Params,
                 training_set: tf.data.Dataset = None,
                 validation_set: tf.data.Dataset = None,
                 test_set: tf.data.Dataset = None):
        """
        Builds a keras model and sets the train and test sets
        :param test_set: tf.Dataset object with images for prediction
        :param training_set: tf.Dataset object with train images
        """

        # Set up the sets and params
        self._test_set = test_set
        self._training_set = training_set
        self._validation_set = validation_set
        self._params = params
        self._model_type = model_type

        # Set a boolean flag to signal that the model has been trained
        self.__trained = False

        # Set the path to the current experiment
        self.__current_experiment_path = os.path.join(os.getcwd(), 'networks', 'experiments', params.run_name)

        # Initialize an empty model
        self._model = models.Sequential()

    def get_params(self) -> Params:
        return self._params

    def get_test_set(self):
        return self._test_set

    def get_model(self) -> models.Sequential:
        return self._model

    def _build(self, params) -> None:
        pass

    def __restore_weights(self, experiment_path, log):
        """
        Restores the weights at the given experiment path
        :param experiment_path: the path to the current experiments, where the weights to be restored reside
        :param log: the logger
        """

        # Set the initial epoch
        if self._params.initial_epoch < 10:
            init_epoch = '0' + str(self._params.initial_epoch) if self._params.initial_epoch < 10 else str(
                self._params.initial_epoch)
        else:
            init_epoch = str(self._params.initial_epoch)

        # Set the regex for the name of the file and its path
        restore_filename_reg = 'weights.{}-*.hdf5'.format(init_epoch)
        restore_path_reg = os.path.join(experiment_path, restore_filename_reg)

        list_files = glob.glob(restore_path_reg)
        assert len(list_files) > 0, 'ERR: No weights file match provided name {}'.format(restore_path_reg)

        # Get the name of the file and its path
        restore_filename = list_files[0].split('/')[-1]
        restore_path = os.path.join(experiment_path, restore_filename)
        assert os.path.isfile(restore_path), 'ERR: Weight file in path {} seems not to be a file'.format(restore_path)

        log.info("Restoring weights in file {}...".format(restore_filename))
        self._model.load_weights(restore_path)

    def __compile_model(self) -> float:
        """
        Sets the proper learning rate and compiles the model
        :return: a learning rate
        """

        # Set the learning rate (possibly defaults to  1e-4)
        lr = self._params.learning_rate if self._params.learning_rate > 0.0 else 0.0001

        # # Set a learning rate decay
        # lr = tf.compat.v1.train.natural_exp_decay(
        #     learning_rate=lr,
        #     global_step=self._params.initial_epoch,
        #     decay_rate=self._params.lr_decay,
        #     decay_steps=self._params.n_images * self._params.training_ratio // self._params.batch_size)

        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']

        # Set the optimizer
        self._model.compile(optimizer=optimizers.Adam(lr=lr),
                            loss=loss,
                            metrics=metrics)

        # build model to be more free while construction
        if self._model_type == "2D":
            self._model.build(input_shape=tf.keras.Input((self._params.image_size, self._params.image_size, self._params.in_channels, 2)).shape)
        elif self._model_type == "3D":
            self._model.build(input_shape=tf.keras.Input((self._params.image_size, self._params.image_size, self._params.in_channels, 1, 2)).shape)
        return lr

    def __setup_callbacks(self) -> List:
        """
        Sets up the callbacks for training
        :return: the early stopping schedule, tensorboard data and the checkpointer
        """

        # Create a folder for the model log of the current experiment
        weights_log_path = os.path.join(self.__current_experiment_path, 'weights')

        # Set up the callback to save the best weights after each epoch
        checkpointer = ModelCheckpoint(filepath=os.path.join(weights_log_path,
                                                             'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min')

        # Set up Tensorboard
        tensorboard = TensorBoard(log_dir=os.path.join(self.__current_experiment_path, 'tensorboard'),
                                  write_graph=True,
                                  histogram_freq=0,
                                  write_grads=True,
                                  write_images=False,
                                  batch_size=self._params.batch_size,
                                  update_freq=self._params.batch_size)

        # Set up early stopping to interrupt the training if val_loss is not increasing after n epochs
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=25,
                                       mode='min')

        csv_logger = CSVLogger(os.path.join(self.__current_experiment_path, "training.csv"), append=True)

        return [early_stopping, tensorboard, checkpointer, csv_logger]

    def train(self):
        """
        Compiles and train the model for the specified number of epochs.
        """

        log = logging.getLogger('execution')
        log.info("Training the model...\n")

        # Set the number of epochs
        epochs = self._params.epochs

        # Compile the model
        log.info("Compiling the model...")
        lr = self.__compile_model()

        log.info("Main parameters:")
        log.info("* Number of epochs  :   " + str(epochs))
        log.info("* Base learning rate:   " + str(self._params.learning_rate) + '\n')

        # Create a folder for the model log of the current experiment
        weights_log_path = os.path.join(self.__current_experiment_path, 'weights')
        os.makedirs(weights_log_path, exist_ok=True)

        # Restore weights, if required
        if self._params.restore_weights:
            self.__restore_weights(weights_log_path, logging.getLogger('execution'))
        else:
            if len(os.listdir(weights_log_path)) > 0:
                raise FileExistsError("{} has trained weights."
                                      "Please change run_name or delete existing folder.".format(weights_log_path))

        logging.info("Model compiled successfully!")

        # Display the architecture of the model
        log.info("Architecture of the model:")
        self._model.summary()

        if self._params.epochs > self._params.initial_epoch:
            log.info("Setting up the checkpointer...")
            callbacks = self.__setup_callbacks()

            # Train the model
            log.info("Starting the fitting procedure...")
            self._model.fit(self._training_set,
                            epochs=epochs,
                            steps_per_epoch=self._params.training_size // self._params.batch_size,
                            validation_data=self._validation_set,
                            validation_steps=self._params.validation_size // self._params.batch_size,
                            callbacks=callbacks,
                            initial_epoch=self._params.initial_epoch)

            log.info("Training done successfully!\n")

        # Set up a flag which states that the network is now trained and can be evaluated
        self.__trained = True

    def evaluate(self) -> (float, float):
        """
        Evaluates the model returning loss and accuracy
        :return: two lists of scalars, one for the loss and one for the metrics
        """

        log = logging.getLogger('testing')
        log.info("Evaluating the model...")

        if not self.__trained:
            # Compile the model
            log.info("Compiling the model...")
            self.__compile_model()

            # Create a folder for the model log of the current experiment
            weights_log_path = os.path.join(self.__current_experiment_path, 'weights')
            self.__restore_weights(weights_log_path, logging.getLogger('execution'))

        return self._model.evaluate(self._test_set, steps=self._params.test_size // self._params.batch_size)

    def predict(self, set=None) -> List[float]:
        """
        Performs a prediction over some items
        :return: the numpy array with predictions
        """

        log = logging.getLogger('execution')
        log.info("Predicting...")

        if not self.__trained:
            # Create a folder for the model log of the current experiment
            weights_log_path = os.path.join(self.__current_experiment_path, 'weights')
            self.__restore_weights(weights_log_path, log)

            # Compile the model
            log.info("Compiling the model...")
            self.__compile_model()

        if set is not None:
            test_set = set
            steps = None
        else:
            test_set = self._test_set
            steps = self._params.test_size // self._params.batch_size

        return self._model.predict(test_set, steps=steps, verbose=1)
