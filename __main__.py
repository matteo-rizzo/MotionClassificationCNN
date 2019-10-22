import logging
import os
from typing import Union, Tuple

import tensorflow as tf

from networks.classes.models.Model2D import Model2D, Model2DSiamese
from networks.classes.models.Model3D import Model3D
from networks.classes.others.Dataset_for_couples import Dataset
from networks.classes.others.Params import Params
from networks.classes.specialization.GradCAM import GradCAM
from networks.classes.specialization.ActivationsHeatmap import ActivationsHeatmap
from networks.classes.specialization.LayersRemover import LayersRemover
from networks.classes.specialization.OutputsVisualizer import OutputsVisualizer
from networks.utility_functions.logging import init_loggers, log_configuration, log_metrics

# tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def initialize_dataset(general_params: Params,
                       model_params: Params,
                       log: logging.Logger) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    log.info('Generating the dataset...')

    # Create the input data pipeline
    train_dataset = Dataset(dim=int(general_params.model[0]), path=os.path.join(general_params.training_dataset), params=model_params)

    # Create different test input data pipeline in case of cross-evaluation
    test_dataset = Dataset(dim=int(general_params.model[0]), path=os.path.join(general_params.test_dataset), params=model_params)

    # Split the dataset
    training_set, validation_set, _ = train_dataset.split(model_params)
    _, _, test_set = test_dataset.split(model_params)

    return training_set, validation_set, test_set


def initialize_model(model_type: str,
                     model_params: Params,
                     training_set: tf.data.Dataset,
                     validation_set: tf.data.Dataset,
                     test_set: tf.data.Dataset,
                     log: logging.Logger) -> Union[Model2D, Model3D]:
    log.info('Building the model...')

    # Set up the specified model
    generic_model = Model2DSiamese if model_type == "2D" else Model3D

    # Build the model
    return generic_model(model_type,
                         model_params,
                         training_set,
                         validation_set,
                         test_set)


def main():
    # Load the general and training parameters from json file
    general_params = Params(os.path.join(os.getcwd(), 'networks', 'config', 'general_params.json'))
    model_params = Params(
        os.path.join(os.getcwd(), 'networks', 'config', 'params_model' + general_params.model + '.json'))

    # Initialize the logger
    run_name = init_loggers(model_params.run_name)
    log = logging.getLogger('execution')

    # Log config
    log.info('Software versions:')
    log.info('* Tensorflow version: ' + tf.__version__)
    log.info('* Keras version:      ' + tf.__version__)
    log.info('* Eager execution:    ' + ('On' if tf.executing_eagerly() else 'Off') + '\n')
    log.info('General parameters:')
    log.info('* Training dataset: ' + general_params.training_dataset)
    log.info('* Test dataset:     ' + general_params.test_dataset)
    log.info('* Dataset type:     ' + general_params.model + '\n')

    # Log general and training parameters
    log_configuration(run_name, general_params.model)

    # Initialize the dataset
    training_set, validation_set, test_set = initialize_dataset(general_params, model_params, log)

    # Initialize the specified model
    model = initialize_model(general_params.model, model_params, training_set, validation_set, test_set,
                             log)

    # Train the model
    if model_params.train:
        log.info('Starting the training procedure...')
        model.train()

    # Evaluate the model
    if model_params.test:
        log.info('Evaluating the model...')
        metrics = model.evaluate()
        log_metrics(metrics, general_params, 'testing')

    if general_params.specialization['run_tests']:

        specialization_tests = general_params.specialization
        test_img = 'dataset/circular_motion_120/coherent/t105-i1_120_coherent.jpg'
        # test_img = 'dataset/circular_dipoles_120/noise/t11-i2_120_noise.jpg'

        # Perform a test on the specialization of the layers
        if specialization_tests['layers_removal']:
            lr = LayersRemover(model.get_model())
            lr.remove_layers({
                'general_params': general_params,
                'log_metrics': log_metrics
            })

        # Plot an heatmap based on the weights of the network
        if specialization_tests['activation_heatmap']:
            h = ActivationsHeatmap(model.get_model())
            h.plot_activation_heatmap(layers='convolutional', pile_layers=True, save=True)

        # Plot a Gradient Class Activation Map (Grad CAM) on a test image
        if specialization_tests['gradcam']:
            gc_handler = GradCAM(model.get_params())
            gc_handler.compute_saliency(img_path=test_img,
                                        cls=0,
                                        layer_name='conv1_2')

        # Plot the intermediate outputs of the convolutional layers
        if specialization_tests['outputs_visualization']:
            ov = OutputsVisualizer(img_path=test_img,
                                   model=model.get_model(),
                                   show_input=False)
            ov.plot_intermediate_outputs(num_layers=50, images_per_row=16, save=False)

    # alessandro_tests(model, model_params, test_set)


def alessandro_tests(model, model_params, test_set):
    # ----- TEST PREDICT ------
    from sklearn.metrics import confusion_matrix, classification_report
    from networks.classes.others.DatasetPrediction import DatasetPrediction
    import numpy as np

    y_true = []
    for _, label in test_set:
        y_true.extend(label.numpy())
    print(y_true)

    prediction = model.predict(test_set)
    y_pred = np.argmax(prediction, axis=1)

    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=['coherent', 'noise']))

    # SECOND TEST

    d = DatasetPrediction(2)
    images = []
    path1 = 'dataset/circular_dipoles_120/noise'
    for image in os.listdir(path1):
        images.append(os.path.join(path1, image))
    path2 = 'dataset/circular_dipoles_120/coherent'
    for image in os.listdir(path2):
        images.append(os.path.join(path2, image))

    ts = d.get_predict_set(model_params, images)
    pred = model.predict(ts)
    y_pred = np.argmax(pred, axis=1)

    y_true = [1] * len(os.listdir(path1))
    y_true.extend([0] * len(os.listdir(path2)))

    print(confusion_matrix(y_true=y_true, y_pred=y_pred))

    # ----- FINE TEST PREDICT ------


def test_dataset():
    from PIL import Image
    import numpy as np
    # Load the general and training parameters from json file
    general_params = Params(os.path.join(os.getcwd(), 'networks', 'config', 'general_params.json'))
    model_params = Params(
        os.path.join(os.getcwd(), 'networks', 'config', 'params_model' + general_params.model + '.json'))

    # Initialize the logger
    run_name = init_loggers(model_params.run_name)
    log = logging.getLogger('execution')

    # Initialize the dataset
    training_set, validation_set, test_set = initialize_dataset(general_params, model_params, log)
    for img, label in training_set:
        img = img.numpy()
        label = label.numpy()
        img = np.array(img[0, :, :, :, 0, 0], np.uint8)*255
        print(np.sum(img))
        Image.fromarray(np.array(img, np.uint8)).show()
        break


if __name__ == '__main__':
    main()
    # test_dataset()
