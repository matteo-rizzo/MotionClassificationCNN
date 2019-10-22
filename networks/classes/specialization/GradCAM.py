import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import image
from matplotlib import pyplot as plt
from tensorflow.python.framework import ops

from networks.classes.models.Model2D import Model2D
from networks.classes.models.Model3D import Model3D
from networks.classes.others import Params
from networks.classes.others.DatasetPrediction import DatasetPrediction

H, W = 150, 150  # Input shape, defined by the model (model.input_shape)


class GradCAM:
    def __init__(self, params: Params, model: str = '2D'):
        self.__params = params
        self.__generic_model = Model2D if model == '2D' else Model3D

    def __load_image(self, path, color_mode='grayscale', reshape=True):
        """
        Loads an image
        """
        if reshape:
            x = image.load_img(path, target_size=(H, W), color_mode=color_mode)
            x = image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
            return x.reshape((150, 150))
        else:
            # Pass '3' to use 3D model, '2' for 2D
            d = DatasetPrediction(2)
            return d.get_predict_set(self.__params, [path])

    @staticmethod
    def __deprocess_image(x, reshape=True):
        """
        Performs image deprocessing and normalization
        """

        x = x.copy()

        if np.ndim(x) > 3:
            x = np.squeeze(x)

        # Normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # Clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # Convert to RGB array
        x *= 255
        if K.image_data_format() == 'th':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')

        return x.reshape((150, 150)) if reshape else x

    def __build_model(self):
        """
        Builds a keras model instance
        """
        self.__model = self.__generic_model(self.__params)
        self.__model.train()
        return self.__model.get_model()

    def __build_guided_model(self):
        """
        Builds a modified model changing gradient function for all ReLu activations
        according to Guided Backpropagation
        """

        if "GuidedBackProp" not in ops._gradient_registry._registry:
            @ops.RegisterGradient("GuidedBackProp")
            def _GuidedBackProp(op, grad):
                dtype = op.inputs[0].dtype
                return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

        with tf.compat.v1.get_default_graph().gradient_override_map({'Relu': 'GuidedBackProp'}):
            return self.__build_model()

    @staticmethod
    def __guided_backprop(input_model, images, layer_name):
        """
        Guided Backpropagation method for visualizing input saliency
        """

        input_imgs = input_model.input
        layer_output = input_model.get_layer(layer_name).output

        backprop_fn = K.function([input_imgs, K.learning_phase()],
                                 [K.gradients(layer_output, input_imgs)[0]])

        return backprop_fn([images, 0])[0]

    @staticmethod
    def __grad_cam(input_model: models.Sequential, img: np.array, cls: int, layer_name: str):
        """
        GradCAM method for visualizing input saliency
        """

        y_c = input_model.output[0, cls]
        conv_output = input_model.get_layer(layer_name).output
        grads = K.gradients(y_c, conv_output)[0]

        # Normalize the gradients by the L2 norm of the vector
        grads = (grads + 1e-10) / (K.sqrt(K.mean(K.square(grads))) + 1e-10)

        gradient_function = K.function([input_model.input], [conv_output, grads])

        output, grads_val = gradient_function([img])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # Process CAM
        cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam_max = cam.max()

        return cam / cam_max if cam_max != 0 else cam

    def __save_plots(self, img_path: str, gradcam, guided_backprop, guided_gradcam):
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + self.__load_image(img_path)) / 2
        cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
        cv2.imwrite('guided_backprop.jpg', self.__deprocess_image(guided_backprop[0]))
        cv2.imwrite('guided_gradcam.jpg', self.__deprocess_image(guided_gradcam[0]))

    def __visualize_plots(self, img_path: str, gradcam, guided_backprop, guided_gradcam):
        img = self.__load_image(img_path)

        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.title('Original')
        plt.imshow(img)

        plt.subplot(222)
        plt.title('GradCAM')
        plt.imshow(img)
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(223)
        plt.title('Guided Backprop')
        plt.imshow(np.flip(self.__deprocess_image(guided_backprop[0]), -1))

        plt.subplot(224)
        plt.title('Guided GradCAM')
        plt.imshow(np.flip(self.__deprocess_image(guided_gradcam[0]), -1))

        plt.show()

    def compute_saliency(self,
                         img_path: str,
                         layer_name: str,
                         cls: int = -1,
                         visualize: bool = True,
                         save: bool = False):
        """
        Computes saliency using all three approaches
        @:param layer_name: layer to compute gradients;
        @:param cls: class number to localize (-1 for most probable class).
        """

        print('\nPerforming Grad CAM analysis...')

        model = self.__build_model()
        guided_model = self.__build_guided_model()

        # Load the input image
        preprocessed_input = self.__load_image(img_path, reshape=False)

        print('Prediction: \n'
              '* Input: {img} \n'
              '* Probabilities: {pred}'.format(img=img_path, pred=model.predict(preprocessed_input)))

        # Convert tf.Dataset to numpy array
        preprocessed_input = tf.data.experimental.get_single_element(preprocessed_input)
        preprocessed_input = preprocessed_input.numpy()

        # Calculate Grad Class Activation Map
        gradcam = self.__grad_cam(model, preprocessed_input, cls, layer_name)

        # Calculate Guided Backpropagation
        guided_backprop = self.__guided_backprop(guided_model, preprocessed_input, layer_name)

        # Calculate Guided Grad CAM mixing the two methods above
        guided_gradcam = guided_backprop * gradcam[..., np.newaxis]

        if save:
            self.__save_plots(img_path, gradcam, guided_backprop, guided_gradcam)

        if visualize:
            self.__visualize_plots(img_path, gradcam, guided_backprop, guided_gradcam)
