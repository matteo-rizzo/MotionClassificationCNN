import threading

from scripts.dataset_generator.image_builders import *
from scripts.dataset_generator.image_writers import *
from scripts.utility.json_handler import Params


def chunks(l: list, n: int):
    """
    Divides a list into n equal parts
    :param l: list
    :param n: number of chunks
    :return: list of list of the original
    """

    # check if n > len(l). In that case we define n as the length of the list
    list_length = len(l)

    if list_length < n:
        n = list_length

    k, m = divmod(len(l), n)

    return (l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class ThreadPerDataset(threading.Thread):
    def __init__(self, name, path_to_dataset_folder, scale, blur, datasets):
        threading.Thread.__init__(self)
        self.name = name
        self.path_to_dataset_folder = path_to_dataset_folder
        self.scale = scale
        self.blur = blur
        self.datasets = datasets

    def run(self) -> None:
        self.write_them_all()

    def write_them_all(self):
        for dataset in self.datasets:

            # "generate" controls whether to regenerate dataset, if False skip dataset
            if not dataset["generate"]:
                continue

            n_images = dataset["n_images"]
            coherency = dataset["coherency"]
            dimension = dataset["dimension"]
            image_type = dataset["image_type"]

            theta = 0.0 if image_type == "circular_dipoles" or image_type == "circular_motion" else dataset["theta"]
            to_write = []

            if isinstance(image_type, list):
                # MIXED DATASET
                # Example: 2D translational motion + circular dipoles = mixed_dataset_2D_tm_cd

                # Compute name for mixed dataset folder
                dir_path = "mixed_{}".format(dimension)
                for itype in image_type:
                    elem = itype.split('_')
                    dir_path += "_" + elem[0][0] + elem[1][0]

                # Print coherency, list or single value
                if isinstance(coherency, list):
                    dir_path += "_" + "_".join(map(str, coherency))
                else:
                    dir_path += "_" + str(coherency)

                results = multiple_builder(types=image_type,
                                           n_images=n_images,
                                           dir_path=dir_path,
                                           path_to_dataset_folder=self.path_to_dataset_folder,
                                           coherent_dipoles=coherency,
                                           theta=theta,
                                           to_write=to_write)
            else:
                # SINGLE TYPE DATASET

                builders = {
                    "circular_dipoles": build_for_circular_dipoles,
                    "translational_dipoles": build_for_translational_dipoles,
                    "circular_motion": build_for_circular_motion,
                    "translational_motion": build_for_translational_motion
                }

                builder = builders[image_type]
                dir_path = image_type if dimension != "3D" else image_type + "_3D"
                dir_path += "_{}".format(coherency)
                dir_path = dir_path if image_type == "circular_dipoles" or image_type == "circular_motion" \
                    else dir_path + "_{}".format(theta)

                radius, max_x, min_x = builder(n_images,
                                               coherency,
                                               to_write,
                                               self.path_to_dataset_folder,
                                               dir_path,
                                               theta)

                results = [{
                    "image_type": image_type,
                    "radius": radius,
                    "max_x": max_x,
                    "min_x": min_x
                }]

            image_data = []
            for data in results:
                max_x = data["max_x"]
                min_x = data["min_x"]
                radius = data["radius"]

                # Image resolution calculation
                img_res = int((abs(max_x) + abs(min_x) + 99) / 100) * 100
                img_res_r = int(img_res / self.scale)
                center = int(img_res / 2)
                img_res = (img_res, img_res)

                print('Images resolution: {}'.format((int(img_res[0] / self.scale)), int(img_res[1] / self.scale)))

                # Set up names for images
                if len(results) == 1:
                    coherent_name = '_{}_coherent.jpg'.format(coherency)
                    noise_name = '_{}_noise.jpg'.format(coherency)
                else:
                    # In mixed dataset mode, omit theta angle for circular dipoles and circular motion
                    tc = data["coherency"]
                    tt = data["theta"]
                    tn = data["image_type"]

                    coherent_name = '_{}_{}_{}_coherent.jpg'.format(tn, tc, tt) \
                        if tn != "circular_dipoles" and tn != "circular_motion" \
                        else '_{}_{}_coherent.jpg'.format(tn, tc)

                    noise_name = '_{}_{}_{}_noise.jpg'.format(tn, tc, tt) \
                        if tn != "circular_dipoles" and tn != "circular_motion" \
                        else '_{}_{}_noise.jpg'.format(tn, tc)

                image_data.append({"radius": radius, "img_res": img_res, "img_res_r": img_res_r,
                                   "center": center, "coherent_name": coherent_name,
                                   "noise_name": noise_name})

            writers = {
                "2D": {
                    "coherent": write_coherent_2D,
                    "noise": write_noise_2D
                },
                "3D": {
                    "coherent": write_coherent_3D,
                    "noise": write_noise_3D
                }
            }

            writer_coherent = writers[dimension]["coherent"]
            writer_noise = writers[dimension]["noise"]

            assert len(to_write) == len(image_data), "To write and image data MUST have same length"

            for i_data, w_data in zip(image_data, to_write):
                print('\nNow writing {image_type} data under {path} folder...'.format(
                    image_type=w_data['image_type'],
                    path=w_data['path']))

                writer_coherent(w_data['coherent'],
                                w_data['path'],
                                self.blur,
                                i_data["img_res"],
                                i_data["img_res_r"],
                                i_data["radius"],
                                i_data["center"],
                                i_data["coherent_name"])

                writer_noise(w_data['noise'],
                             w_data['path'],
                             self.blur,
                             i_data["img_res"],
                             i_data["img_res_r"],
                             i_data["radius"],
                             i_data["center"],
                             i_data["noise_name"])


def write_images_on_disk(parameters):
    path_to_dataset_folder = os.path.join(os.pardir, os.path.join('..', 'dataset'))
    num_threads = parameters.general["num_threads"]

    scale = parameters.general["scale"]
    blur = parameters.general["blur"]

    original_datasets = parameters.datasets

    # Remove every dataset which is not going to be generate
    to_remove = []
    for i, dataset in enumerate(original_datasets):
        if not dataset["generate"]:
            to_remove.append(i)
    for i, idx in enumerate(to_remove):
        idx -= i
        original_datasets.pop(idx)

    # Create threads to write files on disk
    datasets_chunks = chunks(original_datasets, num_threads)
    threads = []

    for i, datasets in enumerate(datasets_chunks):
        threads.append(ThreadPerDataset(str(i), path_to_dataset_folder, scale, blur, datasets))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    # Load the parameters from json file
    json_path = os.path.join(os.getcwd(), os.path.join('params.json'))
    assert os.path.isfile(json_path), "No json config file found at {}".format(json_path)
    params = Params(json_path)

    write_images_on_disk(params)
