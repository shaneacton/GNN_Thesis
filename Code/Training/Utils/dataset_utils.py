import os
from os.path import exists

import nlp


def data_loc(dataset_name, version_name, file_name):
    if not exists(dataset_name):
        print("creating dir:", dataset_name)
        os.mkdir(dataset_name)
    data_name = version_name if version_name else dataset_name
    return os.path.join(data_name, file_name)


def load_unprocessed_dataset(dataset_name, version_name, split):
    """loads the original, unprocessed version of the given dataset"""
    remaining_tries = 100
    dataset = None
    e = None
    while remaining_tries > 0:
        """load dataset from online"""
        try:
            dataset = nlp.load_dataset(path=dataset_name, split=split, name=version_name)
            break  # loaded successfully
        except Exception as e:
            remaining_tries -= 1  # retry
            if remaining_tries == 0:
                print("failed to load datasets though network")
                raise e

    return dataset