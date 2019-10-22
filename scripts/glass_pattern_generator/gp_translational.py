import numpy as np
from scripts.glass_pattern_generator.gp_generator import GPGenerator


class TranslationalDipolesGenerator(GPGenerator):

    def __init__(self, n_images=100, coherent_dipoles=200, theta=45):
        super().__init__(n_images, coherent_dipoles)
        self.__theta2 = theta

    def __noise_frame(self, half=False):

        noise_dipoles = self.noise_dipoles if not half else int(self.num_dots / 2)

        # field 1
        theta1 = self._deg2rad(360 * np.random.rand(noise_dipoles, 1))
        r_adapt = self.min_rad_pix + np.dot((self.max_rad_pix - self.min_rad_pix),
                                            np.sqrt(np.random.rand(noise_dipoles, 1)))
        x1 = np.multiply(r_adapt, np.cos(theta1))
        y1 = np.multiply(r_adapt, np.sin(theta1))
        xy_f1 = np.array([x1, y1])

        # field 2
        theta2 = self._deg2rad(360 * np.random.rand(noise_dipoles, 1))
        x2 = np.multiply(r_adapt, np.cos(theta2))
        y2 = np.multiply(r_adapt, np.sin(theta2))
        xy_f2 = np.array([x2, y2])

        xy_matrix_n = np.concatenate((xy_f1, xy_f2), axis=1).reshape((2, noise_dipoles * 2))

        return xy_matrix_n

    def __coherent_frame(self):

        coherent_dipoles = self.num_dots if self.noise_dipoles == 0 else self.coherent_entities

        # field 1
        theta1 = self._deg2rad(360 * np.random.rand(coherent_dipoles, 1))
        r_adapt = self.min_rad_pix + np.dot((self.max_rad_pix - self.min_rad_pix),
                                            np.sqrt(np.random.rand(coherent_dipoles, 1)))
        x1 = np.multiply(r_adapt, np.cos(theta1))
        y1 = np.multiply(r_adapt, np.sin(theta1))
        xy_f1 = np.array([x1, y1])

        # field 2
        theta2 = self._deg2rad(self.__theta2)
        x2 = x1 + self.dipole_dist_pix * np.cos(theta2)
        y2 = y1 + self.dipole_dist_pix * np.sin(theta2)
        xy_f2 = np.array([x2, y2])

        xy_matrix_c = np.concatenate((xy_f1, xy_f2), axis=1).reshape((2, coherent_dipoles * 2))

        return xy_matrix_c

    def make_images(self):

        for i in range(self.n_images):

            for agp in range(self.interval_duration_frames):
                if self.noise_dipoles == 0:
                    xy_i1 = self.__coherent_frame()
                else:
                    xy_n = self.__noise_frame()
                    xy_c = self.__coherent_frame()
                    xy_i1 = np.concatenate((xy_c, xy_n), axis=1)

                xy_i2 = self.__noise_frame(half=True)

                self.coherent_images[i, :, :, agp] = xy_i1
                self.noise_images[i, :, :, agp] = xy_i2

        print("Size of coherent_images: {i1}\nSize of noise_images: {i2}\n".format(
            i1=self.coherent_images.shape,
            i2=self.noise_images.shape
        ))

        return self.coherent_images, self.noise_images, self.dot_size_pix
