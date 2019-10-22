import numpy as np
from random import shuffle
from scripts.glass_pattern_generator.gp_generator import GPGenerator


class CircularMotionGenerator(GPGenerator):

    def __init__(self, n_images=100, coherent_dipoles=200):
        super().__init__(n_images, coherent_dipoles)

    def __noise_frame(self, num_dots=False):

        noise_dots = self.noise_dots if not num_dots else self.num_dots

        image_matrix = np.zeros((2, noise_dots, 4))

        # first field
        theta1 = self._deg2rad(360 * np.random.rand(noise_dots, 1))
        r_adapt = self.min_rad_pix + np.dot((self.max_rad_pix - self.min_rad_pix),
                                            np.sqrt(np.random.rand(noise_dots, 1)))
        x1 = np.multiply(r_adapt, np.cos(theta1))
        y1 = np.multiply(r_adapt, np.sin(theta1))
        xy_f1 = np.array([x1, y1]).reshape((2, noise_dots))
        image_matrix[:, :, 0] = xy_f1

        ones = np.ones(25)
        zeros = np.full(25, -1)
        directions = np.append(ones, zeros)

        shuffle(directions)

        # following fields
        for fr in range(self.interval_duration_frames)[1:]:
            theta2 = self._deg2rad(360 * directions[fr] * np.random.rand(noise_dots, 1))
            x2 = np.multiply(r_adapt, np.cos(theta2))
            y2 = np.multiply(r_adapt, np.sin(theta2))
            xy_f2 = np.array([x2, y2]).reshape((2, noise_dots))
            image_matrix[:, :, fr] = xy_f2

        return image_matrix

    def __coherent_frame(self):

        coherent_dots = self.num_dots if self.noise_dipoles == 0 else self.coherent_entities

        image_matrix = np.zeros((2, coherent_dots, 4))

        # field 1
        theta1 = self._deg2rad(360 * np.random.rand(coherent_dots, 1))
        r_adapt = self.min_rad_pix + np.dot((self.max_rad_pix - self.min_rad_pix),
                                            np.sqrt(np.random.rand(coherent_dots, 1)))
        x1 = np.multiply(r_adapt, np.cos(theta1))
        y1 = np.multiply(r_adapt, np.sin(theta1))
        xy_f1 = np.array([x1, y1]).reshape((2, coherent_dots))
        image_matrix[:, :, 0] = xy_f1

        # following fields
        for fr in range(self.interval_duration_frames)[1:]:
            theta2 = theta1 + (self.speed_pix / r_adapt)
            x2 = np.multiply(r_adapt, np.cos(theta2))
            y2 = np.multiply(r_adapt, np.sin(theta2))
            xy_f2 = np.array([x2, y2]).reshape((2, coherent_dots))
            image_matrix[:, :, fr] = xy_f2

        return image_matrix

    def make_images(self):

        for i in range(self.n_images):
            if self.noise_dipoles == 0:
                xy_i1 = self.__coherent_frame()
            else:
                xy_n = self.__noise_frame()
                xy_c = self.__coherent_frame()
                xy_i1 = np.concatenate((xy_c, xy_n), axis=1)

            xy_i2 = self.__noise_frame(num_dots=True)

            self.coherent_images[i] = xy_i1
            self.noise_images[i] = xy_i2

        print("Size of coherent_images: {i1}\nSize of noise_images: {i2}\n".format(
            i1=self.coherent_images.shape,
            i2=self.noise_images.shape
        ))

        return self.coherent_images, self.noise_images, self.dot_size_pix
