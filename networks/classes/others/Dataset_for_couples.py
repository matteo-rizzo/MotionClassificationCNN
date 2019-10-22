import os
import random
from typing import List, Tuple

import tensorflow as tf

label_dict = {
    'coherent': 0,
    'noise': 1
}

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, dim: int, path: str, params: any, data: List = None):
        """
        Initializing dataset.
        :param dim: dimension of the images. It may be 2 for 2D conv networks_stanford or 3D for 3D conv networks_stanford
        :param path: path to dataset folder or path to h5 file
        :param data: a dataset made of list of [path, label] values
        """

        self.params = params
        self.__path = path
        self.__data = data
        self.__dim = dim
        self.__train_set = None
        self.__val_set = None
        self.__test_set = None

        self.__build()

    @property
    def train_size(self):
        return len(self.__train_set[0])

    @property
    def val_size(self):
        return len(self.__val_set[0])

    @property
    def test_size(self):
        return len(self.__test_set[0])

    # @tf.function
    def __parse_img(self, couple_images: (str, str), label: int, size: int) -> (tf.Tensor, str):
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
        img_0 = couple_images[0]
        img_1 = couple_images[1]

        def decode_img(img):
            img = tf.io.read_file(img)
            if self.__dim == 2:
                img = tf.image.decode_jpeg(img, channels=1)
            elif self.__dim == 3:
                img = tf.image.decode_png(img, channels=4)
            else:
                raise ValueError('Image dimension not supported')

            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [size, size])

            if self.__dim == 3:
                img = tf.expand_dims(img,
                                     axis=-1)  # insert another dimension at the end of the image (color channels for the 3DConv)
            return img

        img_0 = decode_img(img_0)
        img_1 = decode_img(img_1)

        # stack the two images together
        img = tf.stack((img_0, img_1), axis=-1)

        return img, label

    # def __preprocess_img(self, image: tf.Tensor, label: str, params: any) -> (tf.Tensor, str):
    #     """
    #     Image preprocessing for training.
    #     Apply the following operations:
    #         - Horizontally flip the image with probability 1/2
    #         - Crop image centrally with crop size chosen randomly
    #     :param params: object with parameters
    #     :param image: tensor representing an image
    #     :param label:
    #     :return: the preprocessed image as tensor and the label
    #     """
    #
    #     if params.use_random_flip and not self.__dim == 3:
    #         image = tf.image.random_flip_left_right(image)
    #
    #     # Make sure the image is still in [0, 1]
    #     image = tf.clip_by_value(image, 0.0, 1.0)
    #
    #     return image, label

    # def __fetch_filenames_labels(self,
    #                              params: any,
    #                              get_train: bool = False,
    #                              get_test: bool = False,
    #                              get_validation: bool = False) -> (List[str], List[str]):
    #     """
    #     Fetch all images from folder and apply label. filenames format is a list of paths. Labels a list
    #     of labels.
    #     :param params: object with model parameters (from json file)
    #     :param get_validation: get only validation set
    #     :param get_train: get only training set
    #     :param get_test: get only test set
    #     :return: list of list of filenames, list of corresponding train_labels
    #     """
    #
    #     # Computes indexes of each part of the dataset (train, validation an test)
    #     n_examples = len(self.__data)
    #     idx_train = int(n_examples * params.training_ratio)
    #     idx_validation = int(n_examples * params.validation_ratio) + idx_train
    #     idx_test = int(n_examples * params.test_ratio) + idx_validation
    #
    #     if get_train:
    #         train_data = self.__data[0:idx_train][:]
    #         return list(map(list, zip(*train_data)))[0], list(map(list, zip(*train_data)))[1]
    #     elif get_validation:
    #         validation_data = self.__data[idx_train:idx_validation]
    #         return list(map(list, zip(*validation_data)))[0], list(map(list, zip(*validation_data)))[1]
    #     elif get_test:
    #         test_data = self.__data[idx_validation:idx_test]
    #         return list(map(list, zip(*test_data)))[0], list(map(list, zip(*test_data)))[1]

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

        random.seed(3)
        random.shuffle(co_filenames)
        random.shuffle(no_filenames)
        random.seed(None)

        n_examples = len(co_filenames)
        idx_train = int(n_examples * self.params.training_ratio)
        idx_validation = int(n_examples * self.params.validation_ratio) + idx_train
        idx_test = int(n_examples * self.params.test_ratio) + idx_validation

        self.__train_set = co_filenames[0:idx_train], no_filenames[0:idx_train]

        self.__val_set = co_filenames[idx_train:idx_validation], no_filenames[idx_train:idx_validation]

        self.__test_set = co_filenames[idx_validation:idx_test], no_filenames[idx_validation:idx_test]

    @staticmethod
    def __return_shuffled_lists(dataset: Tuple):
        """
        This methods takes the coherent and noise file names and collects them in order to offer them as couples of images
        (coherent, noise). The (coherent, noise) will correspond to the label "0", the (noise, coherent) to the label "1".
        :param dataset: tuple with (coherent, noise) labels
        :return: filenames, labels as couples
        """
        # retrieve filenames from class
        co_filenames = dataset[0]
        no_filenames = dataset[1]

        # shuffle filenames in order to make mixed couples. This method is intended to be called with
        random.shuffle(co_filenames)
        random.shuffle(no_filenames)
        # create actual labels
        co_labels = [0 for _ in co_filenames]
        no_labels = [1 for _ in no_filenames]
        # wrap filenames and labels together. Now co_f_l are lists of tuples with (filename, label)
        co_f_l = list(zip(co_filenames, co_labels))
        no_f_l = list(zip(no_filenames, no_labels))
        # temp_data will be a list of tuples (((co_filename, 0), (no_filename, 1)), ((no_filename, 0), (co_filename, 1)), ecc)
        temp_data = []
        for i, _ in enumerate(co_f_l):
            r = random.random()
            if r > 0.5:
                temp_data.append((co_f_l[i], no_f_l[i]))
            else:
                temp_data.append((no_f_l[i], co_f_l[i]))
        # now a more simple label is applied
        filenames = []
        labels = []
        for el in temp_data:
            filenames.append((el[0][0], el[1][0]))
            if el[0][1] == 0:  # if the first label is 0 then the couple is (coherent, noise) so we label it as "0"
                # enables the one-hot encoding right here
                labels.append((0, 1))
            else:
                labels.append((1, 0))

        return filenames, labels

    def get_training_set(self, params: any) -> tf.data.Dataset:
        """
        Returns Dataset object containing the training set (samples and labels) with size
        computed according to parameters set in 'params'
        :param params: object with model parameters (from json file)
        :return: the training set as tf Dataset object
        """
        params.training_size = self.train_size

        parse = lambda f, l: self.__parse_img(f, l, self.params.image_size)
        # preprocess = lambda f, l: self.__preprocess_img(f, l, params)

        # # This is the implementation of dataset read directly from images.
        # filenames, labels = self.__return_shuffled_lists(self.__train_set)
        # num_samples = len(filenames)
        # assert len(filenames) == len(labels), "Filenames and labels should have same length"

        # First definition of training_size for params here
        return (tf.data.Dataset.from_tensor_slices(self.__return_shuffled_lists(self.__train_set))
                .shuffle(buffer_size=self.train_size, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                # .map(preprocess, num_parallel_calls=AUTOTUNE)
                .batch(params.batch_size)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_validation_set(self, params: any) -> tf.data.Dataset:
        """
        Returns Dataset object containing the training set (samples and labels) with size
        computed according to parameters set in 'params'
        :param params: object with model parameters (from json file)
        :return: the training set as tf Dataset object
        """
        params.validation_size = self.val_size

        parse = lambda f, l: self.__parse_img(f, l, self.params.image_size)
        # preprocess = lambda f, l: self.__preprocess_img(f, l, params)

        # # This is the implementation of dataset read directly from images.
        # filenames, labels = self.__return_shuffled_lists(self.__train_set)
        # num_samples = len(filenames)
        # assert len(filenames) == len(labels), "Filenames and labels should have same length"

        # First definition of training_size for params here
        return (tf.data.Dataset.from_tensor_slices(self.__return_shuffled_lists(self.__val_set))
                .shuffle(buffer_size=self.val_size, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                # .map(preprocess, num_parallel_calls=AUTOTUNE)
                .batch(params.batch_size)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_test_set(self, params: any) -> tf.data.Dataset:
        """
        Returns Dataset object containing the training set (samples and labels) with size
        computed according to parameters set in 'params'
        :param params: object with model parameters (from json file)
        :return: the training set as tf Dataset object
        """
        params.test_size = self.test_size

        parse = lambda f, l: self.__parse_img(f, l, self.params.image_size)
        # preprocess = lambda f, l: self.__preprocess_img(f, l, params)

        # # This is the implementation of dataset read directly from images.
        # filenames, labels = self.__return_shuffled_lists(self.__train_set)
        # num_samples = len(filenames)
        # assert len(filenames) == len(labels), "Filenames and labels should have same length"

        # First definition of training_size for params here
        return (tf.compat.v1.data.Dataset.from_tensor_slices(self.__return_shuffled_lists(self.__test_set))
                .shuffle(buffer_size=self.test_size, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                # .map(preprocess, num_parallel_calls=AUTOTUNE)
                .batch(params.batch_size)
                .repeat()
                .prefetch(AUTOTUNE))

    def split(self, params: any) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        """
        Split the dataset into training, validation and test sets
        :param params: the parameters to be used in order to split the dataset
        :return: training, validation and test sets
        """
        return self.get_training_set(params), self.get_validation_set(params), self.get_test_set(params)
