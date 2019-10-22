import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DatasetPrediction:
    def __init__(self, dim: int):
        self.__dim = dim

    def __parse_img(self, image: str, size: int) -> (tf.Tensor, str):
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
            # insert another dimension at the end of the image (color channels for the 3DConv)
            img = tf.expand_dims(img, axis=-1)

        return img

    def get_predict_set(self, params, filenames):
        parse = lambda f: self.__parse_img(f, params.image_size)

        params.test_size = len(filenames)
        return (tf.data.Dataset.from_tensor_slices(tf.constant(filenames))
                .map(parse, num_parallel_calls=AUTOTUNE)
                .batch(params.batch_size)
                .prefetch(AUTOTUNE))
