import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import image

from networks.utility_functions.user_check import check_existing_folder


class OutputsVisualizer:

    def __init__(self, img_path: str, model: models.Sequential, show_input: bool = False):
        self.__img = self.__load_image(img_path, show_input)
        self.__model = model

    @staticmethod
    def __load_image(img_path, show) -> np.array:
        # Load the image
        img = image.load_img(img_path, target_size=(150, 150), color_mode='grayscale')

        # Convert the image to an array and preprocess it
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        if show:
            plt.imshow(np.squeeze(img_tensor[0]))
            plt.show()

        return img_tensor

    @staticmethod
    def __display_grid(layer_names: List, activations: List, images_per_row: int, save: bool):
        # Display the feature maps
        for layer_name, layer_activation in zip(layer_names, activations):
            # Number of features in the feature map
            n_features = layer_activation.shape[-1]

            # The feature map has shape (1, size, size, n_features)
            size = layer_activation.shape[1]

            # Set the number of columns in the grid
            n_cols = n_features // images_per_row

            # Tile the activation channels in a matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))

            # Tile each filter into an horizontal grid
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

            scale = 1. / size

            plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

            if save:
                plt.savefig(os.path.join('networks', 'specialization_tests', 'outputs_visualization',
                                         layer_name + '.png'))
            else:
                plt.show()

    def plot_intermediate_outputs(self, num_layers: int = 12, images_per_row: int = 12, save: bool = False):

        print('\nVisualizing the intermediate outputs of the first {} layers...'.format(num_layers))

        if (save and check_existing_folder('outputs_visualization')) or not save:
            # Collect the name of the layers for the plot
            layer_names = [layer.name for layer in self.__model.layers[10:num_layers]]

            # Extract the outputs of the layers
            layer_outputs = [layer.output for layer in self.__model.layers[:num_layers]]

            # Create a model that will return the given outputs on the base of the model input
            activation_model = models.Model(inputs=self.__model.input, outputs=layer_outputs)

            # Perform a prediction on the test image using the new model
            activations = activation_model.predict(self.__img)

            # Display the activations in a grid
            self.__display_grid(layer_names, activations, images_per_row, save)
