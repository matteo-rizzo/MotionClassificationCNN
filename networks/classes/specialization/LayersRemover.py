import copy
from typing import List

import numpy as np
from tensorflow.python.keras import models


class LayersRemover:

    def __init__(self, model: models.Sequential):
        self.__model = model

    def __drop_layer(self, layer: models.Sequential.layers) -> List[np.array]:
        """
        Drops a layer from the model
        :param layer: the layer to be dropped
        :return:
        """
        weights = layer.get_weights()
        zeros = copy.deepcopy(weights)

        for zero in zeros:
            zero.fill(0)

        self.__model.get_layer(layer.name).set_weights(zeros)

        return weights

    def __restore_layer(self, layer: models.Sequential.layers, weights):
        self.__model.get_layer(layer.name).set_weights(weights)

    def remove_layers(self, specialization_dict):
        """
        Performs a test of specialization of the layers removing blocks of layers
        and evaluating the crippled model
        :param specialization_dict: a dictionary with the specifics of the specialization test
        and the related metrics
        """

        # Get the params of the specialization test
        general_params = specialization_dict['general_params']

        # Get the size of the block that must be dropped
        drop_size = int(general_params.specialization['drop_block_size'])
        assert isinstance(drop_size, int) and drop_size > 0, 'ERR: drop_block_size must be a positive integer'

        print('\nTesting specialization of layers with block size {}'.format(drop_size))

        # Get the keras model
        raw_model = self.__model.get_model()

        # Get the list of all layers
        layers_list = raw_model.layers

        # Remove input and output layers
        layers_list.remove(raw_model.get_layer('conv1_0'))
        layers_list.remove(raw_model.get_layer('classifier'))

        # Slide over the list of layers with specified window size
        blocks_to_drop = [layers_list[i:i + drop_size] for i in range(0, len(layers_list), 1)]

        # Remove each block of layers
        for block in blocks_to_drop:
            removed_names = ', '.join(map(lambda l: str(l.name), block))

            print('Dropping layer{} {}...'.format('s' if len(block) > 1 else '', removed_names))

            # Remove layer in block keeping track of original weights
            original_weights = []
            for layer in block:
                weights = self.__drop_layer(layer)
                original_weights.append({'layer': layer, 'weights': weights})

            # Test the model without layers
            print('Evaluating model after dropping layers...')
            params = self.__model.get_params()
            metrics = self.__model.evaluate(self.__model.get_test_set(),
                                            steps=params.test_size // params.batch_size)

            # Save the evaluation metrics into a dict
            specialization_dict['log_metrics'](metrics, general_params, 'execution')

            # Restore weights in the model
            for obj in original_weights:
                self.__restore_layer(obj['layer'], obj['weights'])
