import os
import shutil
from scripts.glass_pattern_generator.gp_generator import GPGenerator
from scripts.glass_pattern_generator.gp_dipoles import CircularDipolesGenerator
from scripts.glass_pattern_generator.gp_circular_motion import CircularMotionGenerator
from scripts.glass_pattern_generator.gp_translational import TranslationalDipolesGenerator
from scripts.glass_pattern_generator.gp_translational_motion import TranslationalMotionGenerator
from typing import List, Dict


def make_folders(path1, path2):
    """
    Creates folders recursively if not present. IF PRESENT IT DELETES the folder and all its content
    UNDER WINDOWS, IF FOLDER IS OPEN IN EXPLORER, IT CAN RAISE AN EXCEPTION "ACCESS DENIED".
    Please close explorer and run again.
    :param path1: string, path/to/folder
    :param path2: string, path/to/folder
    :return: nothing, it creates folders on disk and, if present, deletes everything inside
    """

    try:
        os.makedirs(path1)
    except OSError:
        shutil.rmtree(path1)
        os.makedirs(path1)
    try:
        os.makedirs(path2)
    except OSError:
        shutil.rmtree(path2)
        os.makedirs(path2)


def generate_images(image_type: str, path: str, to_write: List, gp_generator: GPGenerator):
    """
    Utility function to avoid duplicate code. Called by builders to generate images and update
    to_write objects.
    """
    coherent, noise, dot_size_px = gp_generator.make_images()

    to_write.append({
        'image_type': image_type,
        'coherent': coherent,
        'noise': noise,
        'path': path
    })

    # Defining the perfect resolution to fit the images, rounded to the nearest 50th.
    # For most cases it is 1100, we can consider to fix it to that dimension.
    max_x = coherent[0][0, :, 0].max()
    min_x = coherent[0][0, :, 0].min()

    return dot_size_px, max_x, min_x


def multiple_builder(types: List, **kwargs) -> List[Dict]:
    """
    Build multiple types of images.
    :param types: types of images to generate, like "circular_dipoles", "translational_dipoles"
    :param kwargs: dictionary with all relevant params passed to builder. See below functions for
                   reference
    :return: dict with "dot_size_px, max_x, min_x" values indexed by image type
    """
    # Extract parameters
    dir_path = kwargs["dir_path"]
    path_to_dataset_folder = kwargs["path_to_dataset_folder"]
    n_images = kwargs["n_images"]
    coherent_dipoles = kwargs["coherent_dipoles"]
    theta = kwargs["theta"]
    to_write = kwargs["to_write"]
    n = len(types)

    # Make folder for dataset
    coherent_path = os.path.join(path_to_dataset_folder, dir_path, 'coherent')
    noise_path = os.path.join(path_to_dataset_folder, dir_path, 'noise')

    make_folders(coherent_path, noise_path)

    # List to save results
    results = []

    # Deals with lists of parameters or single parameters
    if not isinstance(n_images, list):
        n_images = [n_images] * n

    if not isinstance(coherent_dipoles, list):
        coherent_dipoles = [coherent_dipoles] * n

    if not isinstance(theta, list):
        theta = [theta] * n

    assert len(theta) == n, "In mixed datasets list of params must have same length"
    assert len(n_images) == n, "In mixed datasets list of params must have same length"
    assert len(coherent_dipoles) == n, "In mixed datasets list of params must have same length"

    # Iterate over image types and call the respective generator
    for ty, ni, cohe, the in zip(types, n_images, coherent_dipoles, theta):
        if ty == "circular_dipoles":
            gp_generator = CircularDipolesGenerator(
                n_images=ni,
                coherent_dipoles=cohe)
            the = 0  # Theta angle is not meaningful

        elif ty == "translational_dipoles":
            gp_generator = TranslationalDipolesGenerator(
                n_images=ni,
                coherent_dipoles=cohe,
                theta=the)

        elif ty == "circular_motion":
            gp_generator = CircularMotionGenerator(
                n_images=ni,
                coherent_dipoles=cohe)
            the = 0

        elif ty == "translational_motion":
            gp_generator = TranslationalMotionGenerator(
                n_images=ni,
                coherent_dipoles=cohe,
                theta=the)
        else:
            raise ValueError("Type of image is not valid: \"{}\" is unknown.".format(ty))

        dot_size_px, max_x, min_x = \
            generate_images(ty, os.path.join(path_to_dataset_folder,
                                             dir_path), to_write, gp_generator)
        results.append({
            "image_type": ty,
            "radius": dot_size_px,
            "max_x": max_x,
            "min_x": min_x,
            "coherency": cohe,
            "theta": the})

    return results


def build_for_translational_dipoles(n_images, coherent_dipoles, to_write, path_to_dataset_folder,
                                    dir_path, theta):
    gp_generator = TranslationalDipolesGenerator(
        n_images=n_images,
        coherent_dipoles=coherent_dipoles,
        theta=theta)

    dot_size_px, max_x, min_x = \
        generate_images("translational_dipoles", os.path.join(path_to_dataset_folder, dir_path),
                        to_write, gp_generator)

    coherent_translational_dipoles_path = os.path.join(path_to_dataset_folder, dir_path, 'coherent')
    noise_translational_dipoles_path = os.path.join(path_to_dataset_folder, dir_path, 'noise')

    # Creates folders recursively if not present. IF PRESENT IT DELETES the folder and all its content
    make_folders(coherent_translational_dipoles_path, noise_translational_dipoles_path)

    return dot_size_px, max_x, min_x


def build_for_circular_dipoles(n_images, coherent_dipoles, to_write, path_to_dataset_folder, dir_path,
                               _):
    gp_generator = CircularDipolesGenerator(
        n_images=n_images,
        coherent_dipoles=coherent_dipoles)

    dot_size_px, max_x, min_x = \
        generate_images("circular_dipoles", os.path.join(path_to_dataset_folder, dir_path),
                        to_write, gp_generator)

    coherent_circular_dipoles_path = os.path.join(path_to_dataset_folder, dir_path, 'coherent')
    noise_circular_dipoles_path = os.path.join(path_to_dataset_folder, dir_path, 'noise')

    # Creates folders recursively if not present. IF PRESENT IT DELETES the folder and all its content
    make_folders(coherent_circular_dipoles_path, noise_circular_dipoles_path)

    return dot_size_px, max_x, min_x


def build_for_circular_motion(n_images, coherent_dipoles, to_write, path_to_dataset_folder, dir_path, _):
    gp_generator = CircularMotionGenerator(
        n_images=n_images,
        coherent_dipoles=coherent_dipoles)

    dot_size_px, max_x, min_x = \
        generate_images("circular_motion", os.path.join(path_to_dataset_folder, dir_path),
                        to_write, gp_generator)

    coherent_circular_motion_path = os.path.join(path_to_dataset_folder, dir_path, 'coherent')
    noise_circular_motion_path = os.path.join(path_to_dataset_folder, dir_path, 'noise')

    # Creates folders recursively if not present. IF PRESENT IT DELETES the folder and all its content
    make_folders(coherent_circular_motion_path, noise_circular_motion_path)

    return dot_size_px, max_x, min_x


def build_for_translational_motion(n_images, coherent_dipoles, to_write, path_to_dataset_folder,
                                   dir_path, theta):
    gp_generator = TranslationalMotionGenerator(
        n_images=n_images,
        coherent_dipoles=coherent_dipoles,
        theta=theta)

    dot_size_px, max_x, min_x = \
        generate_images("translational_motion", os.path.join(path_to_dataset_folder, dir_path),
                        to_write, gp_generator)

    coherent_translational_motion_path = os.path.join(path_to_dataset_folder, dir_path, 'coherent')
    noise_translational_motion_path = os.path.join(path_to_dataset_folder, dir_path, 'noise')

    # Creates folders recursively if not present. IF PRESENT IT DELETES the folder and all its content
    make_folders(coherent_translational_motion_path, noise_translational_motion_path)

    return dot_size_px, max_x, min_x
