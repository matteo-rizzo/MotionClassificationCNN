import os


def check_existing_folder(dir_name: str) -> bool:
    """
    Check if the provided directory is empty and asks the user for permission to regenerate it
    :param dir_name: the name of the folder to be regenerated
    :return: the user input converted to bool
    """

    if os.path.isdir(os.path.join('networks', 'specialization_tests', dir_name)):
        user_input = input('\nThe {} folder is not empty and will be overwritten.'
                           ' Continue? [Y/N]\n'.format(dir_name))

        if user_input in ['Y', 'y', 'ok', 'sure', 'yes']:
            os.remove(os.path.join('networks', 'specialization_tests', dir_name))
        else:
            print('Execution of activation heatmaps plot aborted')
            return False

    os.makedirs(os.path.join('networks', 'specialization_tests', dir_name), exist_ok=True)
    return True
