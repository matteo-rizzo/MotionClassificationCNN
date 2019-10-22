import os
import random
from typing import List

import tensorflow as tf

label_dict = {
    'coherent': 0,
    'noise': 1
}

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, dim: int, path: str, params: any = None, data: List = None):
        """
        Initializing dataset.
        :param dim: dimension of the images. It may be 2 for 2D conv networks_stanford or 3D for 3D conv networks_stanford
        :param path: path to dataset folder or path to h5 file
        :param data: a dataset made of list of [path, label] values
        """

        self.__path = path
        self.__data = data
        self.__dim = dim

        self.__build()

    def __parse_img(self, image: str, label: str, size: int) -> (tf.Tensor, str):
        """
        Obtain the image from the filename (for both training and validation).
        The following operations are applied:
            - Decode the image from jpeg format
            - Convert to float and to range [0, 1]
            - Resize image
        :param image: path of the image file
        :param label: string with image label
        :param size: image size for resize operation
        :return: image and its label
        """

        img = tf.io.read_file(image)
        if self.__dim == 2:
            img = tf.image.decode_jpeg(img, channels=1)
        elif self.__dim == 3:
            img = tf.image.decode_png(img, channels=4)
        else:
            raise ValueError('Image dimension not supported')

        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [size, size])

        if self.__dim == 3:
            # Insert another dimension at the end of the image (color channels for the 3DConv)
            img = tf.expand_dims(img, axis=-1)

        return img, label

    def __preprocess_img(self, image: tf.Tensor, label: str, params: any) -> (tf.Tensor, str):
        """
        Image preprocessing for training.
        Apply the following operations:
            - Horizontally flip the image with probability 1/2
            - Crop image centrally with crop size chosen randomly
        :param params: object with parameters
        :param image: tensor representing an image
        :param label:
        :return: the preprocessed image as tensor and the label
        """

        if params.use_random_flip and not self.__dim == 3:
            image = tf.image.random_flip_left_right(image)

        # Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def __fetch_filenames_labels(self,
                                 params: any,
                                 get_train: bool = False,
                                 get_test: bool = False,
                                 get_validation: bool = False) -> (List[str], List[str]):
        """
        Fetch all images from folder and apply label. filenames format is a list of paths. Labels a list
        of labels.
        :param params: object with model parameters (from json file)
        :param get_validation: get only validation set
        :param get_train: get only training set
        :param get_test: get only test set
        :return: list of list of filenames, list of corresponding train_labels
        """

        # Computes indexes of each part of the dataset (train, validation an test)
        n_examples = len(self.__data)
        idx_train = int(n_examples * params.training_ratio)
        idx_validation = int(n_examples * params.validation_ratio) + idx_train
        idx_test = int(n_examples * params.test_ratio) + idx_validation

        if get_train:
            train_data = self.__data[0:idx_train][:]
            return list(map(list, zip(*train_data)))[0], list(map(list, zip(*train_data)))[1]
        elif get_validation:
            validation_data = self.__data[idx_train:idx_validation]
            return list(map(list, zip(*validation_data)))[0], list(map(list, zip(*validation_data)))[1]
        elif get_test:
            test_data = self.__data[idx_validation:idx_test]
            return list(map(list, zip(*test_data)))[0], list(map(list, zip(*test_data)))[1]

    def __build(self):
        """
        Creates dataset from disk. It sets <self.data> as a random list of [path, label] to images
        """

        if os.listdir(self.__path) == 0:
            raise FileNotFoundError('The specified folder does not contain any image.'
                                    ' Please double check dataset folder')

        # Initialize an empty list of coherent images filenames
        co_filenames = []

        # Initialize an empty list of noise images filenames
        no_filenames = []

        for dirpath, dirnames, filenames in os.walk(os.path.join(self.__path, 'coherent')):
            for filename in sorted(filenames):
                co_filenames.append(os.path.join(os.getcwd(), dirpath, filename))

        for dirpath, dirnames, filenames in os.walk(os.path.join(self.__path, 'noise')):
            for filename in sorted(filenames):
                no_filenames.append(os.path.join(os.getcwd(), dirpath, filename))

        co_labels = [label_dict['coherent'] for _ in co_filenames]
        no_labels = [label_dict['noise'] for _ in no_filenames]

        # add a new column with labels
        co_filenames.extend(no_filenames)
        co_labels.extend(no_labels)

        self.__data = list(map(list, zip(*[co_filenames, co_labels])))
        self.__shuffle(3)

    def __shuffle(self, seed: int = int(random.random() * 100)):
        """
        Shuffles dataset with seed, so multiple calls to this method can shuffle the dataset in the same way
        :param seed: seed of shuffle
        """
        random.Random(seed).shuffle(self.__data)

    def get_training_set(self, params: any) -> tf.data.Dataset:
        """
        Returns Dataset object containing the training set (samples and labels) with size
        computed according to parameters set in 'params'
        :param params: object with model parameters (from json file)
        :return: the training set as tf Dataset object
        """

        parse = lambda f, l: self.__parse_img(f, l, params.image_size)
        preprocess = lambda f, l: self.__preprocess_img(f, l, params)

        # This is the implementation of dataset read directly from images.
        filenames, labels = self.__fetch_filenames_labels(get_train=True, params=params)
        num_samples = len(filenames)
        assert len(filenames) == len(labels), "Filenames and labels should have same length"

        # First definition of training_size for params here
        params.training_size = num_samples
        return (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                .map(preprocess, num_parallel_calls=AUTOTUNE)
                .batch(params.batch_size)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_validation_set(self, params: any) -> tf.data.Dataset:
        """
        Returns Dataset object containing the validation set (samples and labels) with size
        computed according to parameters set in 'params'
        :param params: object with model parameters (from json file)
        :return: the validation set as tf Dataset object
        """

        parse = lambda f, l: self.__parse_img(f, l, params.image_size)
        filenames, labels = self.__fetch_filenames_labels(get_validation=True, params=params)
        assert len(filenames) == len(labels), "Filenames and labels should have same length"

        params.validation_size = len(filenames)
        return (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(buffer_size=params.validation_size, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                .batch(params.batch_size)
                .prefetch(AUTOTUNE))

    def get_test_set(self, params: any) -> tf.data.Dataset:
        """
        Returns Dataset object containing the test set (samples and labels) with size
        computed according to parameters set in 'params'
        :param params: object with model parameters (from json file)
        :return: the test set as tf Dataset object
        """

        parse = lambda f, l: self.__parse_img(f, l, params.image_size)

        filenames, labels = self.__fetch_filenames_labels(get_test=True, params=params)
        assert len(filenames) == len(labels), "Filenames and labels should have same length"

        params.test_size = len(filenames)
        return (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(buffer_size=params.test_size, reshuffle_each_iteration=False)
                .map(parse, num_parallel_calls=AUTOTUNE)
                .batch(params.batch_size)
                .prefetch(AUTOTUNE))

    def split(self, params: any) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        """
        Split the dataset into training, validation and test sets
        :param params: the parameters to be used in order to split the dataset
        :return: training, validation and test sets
        """
        return self.get_training_set(params), self.get_validation_set(params), self.get_test_set(params)
