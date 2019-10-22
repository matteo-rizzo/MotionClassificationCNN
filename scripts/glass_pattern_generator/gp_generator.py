import numpy as np
import math


class GPGenerator:

    @staticmethod
    def _deg2rad(degrees):
        """
        Converts degrees into radiant
        :param degrees: a value in degree
        :return: the given value converted into radiant
        """

        return degrees * math.pi / 180

    @staticmethod
    def __visual_angle(params):
        """
        Calculates the visual angle subtended by a single pixel.
        :param params: monitor parameters involving:
            res - the resolution of the monitor
            sz - the size of the monitor in cm
            vd - viewing distance in cm
        (these values can either be along a single dimension or for both the width and height)
        :return pix_per_deg, deg_per_pix: the pixels per degree and it's reciprocal - the degrees per pixel
        (in degrees, not radians)
        """

        v_dist = params['vd']
        pix = np.divide(params['sz'], params['res'])  # calculates the size of a pixel in cm
        deg_per_pix = (2 * np.arctan(np.divide(pix, (2 * v_dist)))) * (180 / math.pi)
        pix_per_deg = 1 / deg_per_pix

        return pix_per_deg, deg_per_pix

    def __noise_frame(self):
        pass

    def __coherent_frame(self):
        pass

    def make_images(self):
        pass

    def __init__(self, n_images=100, coherent_entities=200):
        """
        :param n_images: total number of trial to be executed
        :param coherent_entities: number of coherent dipoles in the image
        :return: coherent_images, noise_images, dot_size_pix
        """

        self.n_images = n_images  # number of trial to be executed

        self.num_dipoles = 688  # total number of dipoles
        self.num_dots = self.num_dipoles * 2  # total number of dots

        self.coherent_entities = coherent_entities  # number of coherent entities (dots or dipoles)

        self.noise_dipoles = self.num_dipoles - self.coherent_entities  # number of noisy dipoles
        self.noise_dots = self.num_dots - self.coherent_entities  # number of noisy dots

        # monitor parameters
        vsonic_p = {
            'res': [1920, 1080],  # monitor resolution
            'sz': [35, 26],  # monitor size in cm
            'vd': 125  # viewing distance in cm
        }

        pix_per_deg, _ = self.__visual_angle(vsonic_p)  # pixel per degree
        pix_deg = pix_per_deg[0]

        speed = 0.30  # 0.18 deg
        self.speed_pix = speed * pix_deg

        dot_width = 0.04  # dot width (deg)
        dipole_dist = 0.18  # deg

        max_rad_deg = 4.5  # maximum radius of the stimulus(deg)
        min_rad_deg = 0.5  # minimum radius of the stimulus(deg)

        self.max_rad_pix = max_rad_deg * pix_deg  # maximum radius of annulus (pixels from center)
        self.min_rad_pix = min_rad_deg * pix_deg  # minimum radius of the stimulus (deg)

        self.dot_size_pix = dot_width * pix_deg  # dot size (pixels)
        self.dipole_dist_pix = dipole_dist * pix_deg

        self.interval_duration_frames = 4

        # x, y coordinates of the first frame of the video pattern
        self.coherent_images = np.zeros((self.n_images, 2, self.num_dots, self.interval_duration_frames))
        # x, y coordinates of the second frame of the video pattern
        self.noise_images = np.zeros((self.n_images, 2, self.num_dots, self.interval_duration_frames))
