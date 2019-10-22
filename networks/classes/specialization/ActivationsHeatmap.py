import os
import re
from typing import List, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from networks.utility_functions.user_check import check_existing_folder


class ActivationsHeatmap:

    def __init__(self, model):
        self.__model = model
        self.__save = False

    def __set_save(self, save):
        self.__save = save

    def __plot_heatmap(self, layer: str, data: np.array, row_labels: range = None, col_labels: range = None):
        """
        Visualizes the heatmap
        :param layer: the name of the current layer (to be plotted in the title)
        :param data: a 2D numpy array of shape (N, M)
        :param row_labels: a list or array of length N with the labels for the rows
        :param col_labels: a list or array of length M with the labels for the columns
        """

        plt.figure()
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(data, cmap='BuPu')

        tick_spacing = 1

        if col_labels:
            ax.set_xticklabels(col_labels)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        if row_labels:
            ax.set_yticklabels(row_labels)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.set_title("Activations of layer {}".format(layer))
        ax.set_xlabel('Hidden unit activation')
        ax.set_ylabel('Layer')

        if self.__save:
            plt.savefig(os.path.join('networks', 'specialization_tests', 'activations_heatmap', layer + '.png'))
        else:
            plt.show()

    def __collapse_list(self, l: List) -> List:
        """
        Recursively collapse a lists of nested lists into a list
        of non nested lists
        :param l: a list of nested lists
        :return: a collapsed list of nested lists
        """
        is_nested = all(isinstance(nested_l, list) for nested_l in l)
        return sum(map(self.__collapse_list, l), []) if is_nested else [l]

    @staticmethod
    def __normalize_weights(l: List, threshold: float) -> np.array:
        """
        For each element of the given list, sets it to zero if
        its value is below a certain threshold
        :param l: the list of weights to normalize
        :return: the normalized list of weights
        """
        return np.array([[0 if item < threshold else item for item in nested_l] for nested_l in l])

    def __plot_layer_heatmap(self, layer_name: str, n_chunks: int, m: int, pile_layers: bool):
        """
        Plots an heatmap for the activations of the given layer
        :param layer_name: the name of the layer whose activations must be plotted
        :param n_chunks: the number of chunks to split the list of weights in
        """
        # Get the list of weights for each layer
        layer_weights = self.__model.get_layer(layer_name).get_weights()[0].tolist()
        if not layer_weights:
            raise Exception('ERR: The provided layer has no weights!')

        # Transform the arbitrarily nested list of lists into a list of non-nested lists
        if pile_layers:
            weights = [np.array(self.__collapse_list(w)).mean(axis=0).tolist() for wl in layer_weights for w in wl]
        else:
            weights = self.__collapse_list(layer_weights)

        # Split the weights into chunks to perform the plot
        weights_chunks = [weights[i:i + n_chunks] for i in range(0, len(weights), n_chunks)]

        # Set the threshold for the normalization of the weights
        threshold = m * np.mean(np.array(weights))

        for i, chunk in enumerate(weights_chunks):
            # Set each element below the mean to zero
            # chunk = self.__normalize_weights(chunk, threshold)

            # Plot the heatmap for the current layer
            self.__plot_heatmap(layer=layer_name + '_' + str(i), data=chunk)

    def plot_activation_heatmap(self,
                                layers: Union[str, List] = 'convolutional',
                                chunks: int = 750,
                                threshold_factor: int = 0,
                                pile_layers: bool = True,
                                save: bool = False):
        """
        Plots an heatmap based on the activation of the neurons of the convolutional layers
        """

        print('\nPlotting activation heatmaps...')

        if (save and check_existing_folder('activations_heatmap')) or not save:

            self.__set_save(save)

            if layers == 'convolutional':
                # Get the names of convolutional layers in the network
                layers_names = [layer.name for layer in self.__model.layers if re.search('conv\d+_\d+', layer.name)]
            else:
                # Use the given layers
                layers_names = [layer.name for layer in self.__model.layers if layer.name in layers]

            # Plot an heatmap for each layer
            for layer in layers_names:
                print('Analyzing layer {}...'.format(layer))
                self.__plot_layer_heatmap(layer_name=layer,
                                          n_chunks=chunks,
                                          m=threshold_factor,
                                          pile_layers=pile_layers)
