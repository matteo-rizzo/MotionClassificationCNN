from scripts.utility.json_handler import Params
import os
from typing import List, Tuple
import random
import h5py
import numpy as np
from PIL import Image, ImageFile
import threading

# force pillow to load also truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

labels = {
    'coherent': 0,
    'noise': 1
}

DATASET_NAME = '4d-misc'


class ThreadedH5pyFile(threading.Thread):
    """
    Threaded version to prepare the dataset. One thread writes odd (1) labelled images, the other the even one
    """

    def __init__(self, img_list: List[str], set_type: str, hdf5_out: h5py.File, mode: str, img_size: int):
        """
        Each thread has one list between coherent and noise
        :param img_list: list of path/to/images
        :param set_type: coherent or noise image type
        :param hdf5_out: shared file in output
        :param img_size: image size
        """
        threading.Thread.__init__(self)
        self.img_list = img_list
        self.set_type = set_type
        self.hdf5_out = hdf5_out
        self.mode = mode
        self.img_size = img_size

    def run(self):
        for i, path in enumerate(self.img_list):
            #  path contains the label and the path_to_image. It's a tuple
            if i % 100 == 0:
                print('\n-------------\nThread: {name}\nSaved images: {c}/{tot}'.format(
                    name=self.mode + "_" + self.set_type,
                    c=i,
                    tot=len(self.img_list)
                ))
            if self.set_type == 'coherent':
                # write in even position
                pos = i * 2
            elif self.set_type == 'noise':
                # write in odd position
                pos = (i * 2) + 1
            else:
                raise ValueError('set_type is not well formatted. Please choose between "coherent" and "noise" types.')
            # some images are not well formatted or have bytes error. So we try to open them and, in case of error,
            # keep the path inside the array self.errors
            img = Image.open(path)
            img = img.resize((self.img_size, self.img_size), Image.ANTIALIAS)
            img_np = np.asarray(img, dtype=np.uint8)
            self.hdf5_out[self.mode + "/images"][pos, ...] = img_np
            self.hdf5_out[self.mode + "/labels"][pos, ...] = labels[self.set_type]

        print('THREAD {} HAS FINISHED'.format(self.mode + "_" + self.set_type))


def images_in_paths(folder_path: str) -> (List[str], List[str]):
    """
    Collects all images from one folder and return a list of paths
    :param folder_path:
    :return:
    """
    co_paths = []
    no_paths = []
    coherent = os.path.join(os.getcwd(), folder_path, 'coherent')
    noise = os.path.join(os.getcwd(), folder_path, 'noise')
    # extensions = set([])
    for root, dirs, files in os.walk(coherent):
        for file in files:
            # extensions.add(os.path.splitext(file)[1])
            co_paths.append(os.path.join(root, file))
    for root, dirs, files in os.walk(noise):
        for file in files:
            # extensions.add(os.path.splitext(file)[1])
            no_paths.append(os.path.join(root, file))
    # with open('extensions.txt', 'w') as f:
    #     f.writelines([extension + '\n' for extension in extensions])
    return co_paths, no_paths


def shuffle_dataset(lst: List, seed: int = None) -> None:
    """
    Controlled shuffle.
    :param lst:
    :param seed: if specified the shuffle returns the same shuffled list every time it is invoked
    :return:
    """
    if seed is not None:
        random.seed(seed)
    random.shuffle(lst)


def generate_h5py(
        file_list: Tuple[List[str], List[str]],
        img_size: int = 256,
        in_channels: int = 4,
        hdf5_file_name: str = 'data',
        folder: str = 'h5_files',
        train_ratio: float = 0.6,
        val_ratio: float = 0.3,
        test_ratio: float = 0.1):
    """
    Generate and save images in h5 file. Since there are 2 classes, the maximum random distribution is [0, 1, 0, 1, ...]
    :param file_list:
    :param img_size:
    :param train_dim:
    :param val_dim:
    :param hdf5_file_name:
    :return:
    """
    co_list = file_list[0]
    no_list = file_list[1]
    # make train, validation and test partitions. num_samples is the length of coherent only, so then we multiply per 2
    num_samples = len(file_list[0])
    # we need to keep odd dimension to keep 1 for coeherent and one for noise
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = int(num_samples * test_ratio)

    co_train = co_list[0:n_train]
    no_train = no_list[0:n_train]
    co_val = co_list[n_train:n_train + n_val]
    no_val = no_list[n_train:n_train + n_val]
    co_test = co_list[n_train + n_val:-1]
    no_test = no_list[n_train + n_val:-1]

    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(os.getcwd(), folder, hdf5_file_name), mode='w') as hdf5_out:
        hdf5_out.create_dataset('train/images', (n_train * 2, img_size, img_size, in_channels), np.uint8)
        hdf5_out.create_dataset('train/labels', (n_train * 2, 1), np.uint8)
        hdf5_out.create_dataset('train/num', (), np.uint32, data=n_train * 2)
        hdf5_out.create_dataset('val/images', (n_val * 2, img_size, img_size, in_channels), np.uint8)
        hdf5_out.create_dataset('val/labels', (n_val * 2, 1), np.uint8)
        hdf5_out.create_dataset('val/num', (), np.uint32, data=n_val * 2)
        hdf5_out.create_dataset('test/images', (n_test * 2, img_size, img_size, in_channels), np.uint8)
        hdf5_out.create_dataset('test/labels', (n_test * 2, 1), np.uint8)
        hdf5_out.create_dataset('test/num', (), np.uint32, data=n_test * 2)

        # make one thread for coherent and noise and mode
        train_co = ThreadedH5pyFile(co_train, 'coherent', hdf5_out, 'train', img_size)
        train_no = ThreadedH5pyFile(no_train, 'noise', hdf5_out, 'train', img_size)
        val_co = ThreadedH5pyFile(co_val, 'coherent', hdf5_out, 'val', img_size)
        val_no = ThreadedH5pyFile(no_val, 'noise', hdf5_out, 'val', img_size)
        test_co = ThreadedH5pyFile(co_test, 'coherent', hdf5_out, 'test', img_size)
        test_no = ThreadedH5pyFile(no_test, 'noise', hdf5_out, 'test', img_size)

        train_co.start()
        train_no.start()
        val_co.start()
        val_no.start()
        test_co.start()
        test_no.start()

        # wait until all threads has finished, so the h5file is kept open
        train_co.join()
        train_no.join()
        val_co.join()
        val_no.join()
        test_co.join()
        test_no.join()


if __name__ == '__main__':
    # Load the parameters from json file
    json_path = os.path.join(os.getcwd(), os.path.join('params.json'))
    assert os.path.isfile(json_path), "No json config file found at {}".format(json_path)
    params = Params(json_path)

    output_path = os.path.join(os.getcwd(), os.pardir, os.pardir, 'dataset', DATASET_NAME)
    # elements = int(1e5)  # number of images to keep
    res_path = os.path.join(os.getcwd(), os.pardir, os.pardir, 'dataset', DATASET_NAME)
    lists = images_in_paths(os.path.join(res_path))
    generate_h5py(lists, params.image_size, params.in_channels, DATASET_NAME + '.h5', folder=output_path)
