import json


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file.
    :param d: (dict) of float-castable values (np.float, int, float, etc.)
    :param json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


class Params:
    """
    Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """
        Saves parameters to json file.
        :param json_path: the path to the json file
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Loads parameters from json file.
        :param json_path: the path to the json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Gives dict-like access to Params instance by `params.dict['learning_rate']`.
        """
        return self.__dict__
