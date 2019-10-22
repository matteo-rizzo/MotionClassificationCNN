import logging
import os
import sys
from shutil import copy


def init_loggers(run_name) -> str:
    """
    Initialize the loggers.
    """

    # Set a format
    frm = '%(asctime)s - %(levelname)s - %(message)s'

    # Set the path to the experiments folder
    experiments_path = os.path.join(os.getcwd(), "networks", "experiments")
    os.makedirs(experiments_path, exist_ok=True)

    # Set up a new directory for the current experiment
    log_directory_name = run_name
    log_directory_path = os.path.join(experiments_path, log_directory_name)
    os.makedirs(log_directory_path, exist_ok=True)

    # Create a logger for the execution
    exec_log_path = os.path.join(log_directory_path, 'execution.log')
    exec_logger = logging.getLogger('execution')
    exec_logger.setLevel('INFO')

    fh = logging.FileHandler(exec_log_path, mode='w')
    fh.setFormatter(logging.Formatter(frm))
    exec_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(frm))
    exec_logger.addHandler(sh)

    # Create a logger for the training
    train_log_path = os.path.join(log_directory_path, 'training.log')
    train_logger = logging.getLogger('training')
    train_logger.setLevel('INFO')

    fh = logging.FileHandler(train_log_path, mode='a')
    fh.setFormatter(logging.Formatter(frm))
    train_logger.addHandler(fh)

    # Create a logger for the testing
    test_log_path = os.path.join(log_directory_path, 'test.log')
    test_logger = logging.getLogger('testing')
    test_logger.setLevel('INFO')
    fh = logging.FileHandler(test_log_path, mode='a')
    fh.setFormatter(logging.Formatter(frm))
    test_logger.addHandler(fh)

    return run_name


def log_configuration(run_name, model):
    """
    Log the parameters json files for the current experiment creating a copy of them.
    :param run_name: the identification code for the current experiment
    :param model: the selected model (i.e. 2D or 3D)
    """
    # Path to the config folder
    config_path = os.path.join(os.getcwd(), 'networks', 'config')

    # Path to classes
    class_path = os.path.join(os.getcwd(), 'networks', 'classes', 'models')

    # Path to the log of the current experiment
    experiment_path = os.path.join(os.getcwd(), 'networks', 'experiments', run_name)

    # Path to the config log for the current experiment
    config_log_path = os.path.join(experiment_path, 'config')
    os.makedirs(config_log_path, exist_ok=True)

    # Log general parameters
    copy(os.path.join(config_path, 'general_params.json'), config_log_path)

    # Log model parameters
    copy(os.path.join(config_path, 'params_model' + model + '.json'), config_log_path)

    # Log network optimizers, losses and metrics
    copy(os.path.join(class_path, 'Model.py'), config_log_path)

    # Log network architecture
    copy(os.path.join(class_path, 'Model' + model + '.py'), config_log_path)


def log_metrics(metrics, params, log_type):
    """
    Log the loss and accuracy metrics for the current experiment.
    :param params: general params object
    :param metrics: loss and accuracy for the current experiment
    :param log_type: the type of logger to be used
    """

    log = logging.getLogger(log_type)

    log.info('Test set: ' + str(params.test_dataset))
    log.info('Metrics:')
    log.info(' * Loss:        ' + str(metrics[0]))
    log.info(' * Accuracy:    ' + str(metrics[1]) + '\n')
