import os
from os.path import exists

import torch
from torch.utils.data import DataLoader

import nlp

from Code.Training.Utils.initialiser import get_tokenizer
from Code.Training.Utils.text_encoder import TextEncoder


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


def load_processed_datasets(dataset_name, version_name, train_split_name, valid_split_name):
    """must call process first with corresponding train/valid_split_name"""
    train_dataset = torch.load(data_loc(dataset_name, version_name, train_split_name))
    valid_dataset = torch.load(data_loc(dataset_name, version_name, valid_split_name))

    return train_dataset, valid_dataset


def get_latest_model(dataset_name, version_name, model_folder_name):
    """finds the model checkpoint which did the most training iterations, returns file name"""
    # out = os.path.join("..", "..", data_loc(dataset_name, version_name, model_folder_name))
    out = data_loc(dataset_name, version_name, model_folder_name)
    checks = [c for c in os.listdir(out) if "check" in c]
    if len(checks) == 0:
        return None
    steps = [int(c.split("-")[1]) for c in checks]
    hi=-1
    max_i = -1
    for i in range(len(steps)):
        if steps[i] > hi:
            hi=steps[i]
            max_i = i
    return checks[max_i]


def get_processed_data_sample(dataset_name, version_name, train_split_name):
    """returns the first data sample from the given preprocessed and saved dataset"""
    dataset = torch.load(data_loc(dataset_name, version_name, train_split_name))
    dataloader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for example in nlp.tqdm(dataloader):
            return example


def process_gat_dataset(dataset_name, version_name, train_split_name, valid_split_name):
    valid_data_loc = data_loc(dataset_name, version_name, valid_split_name)
    if exists(valid_data_loc):
        """already saved"""
        return
    encoder = TextEncoder(get_tokenizer())
    map_func = encoder.get_processed_example
    train_dataset, valid_dataset = _get_processed_dataset(map_func, dataset_name, version_name)

    dataloader = DataLoader(valid_dataset, batch_size=1)
    sample = None
    for sample in nlp.tqdm(dataloader):
        # print("after", batch)
        break

    # set the tensor type and the columns which the dataset should return
    if 'start_positions' in sample and 'end_positions' in sample:
        tensor_columns = ['start_positions', "end_positions"]
    elif "answer" in sample:
        tensor_columns = ['answer']
    else:
        raise Exception()

    train_dataset.set_format(type='torch', columns=tensor_columns, output_all_columns=True)
    valid_dataset.set_format(type='torch', columns=tensor_columns, output_all_columns=True)

    torch.save(train_dataset, data_loc(dataset_name, version_name, train_split_name))
    torch.save(valid_dataset, valid_data_loc)


def process_span_dataset(dataset_name, version_name, train_split_name, valid_split_name):
    valid_data_loc = data_loc(dataset_name, version_name, valid_split_name)
    if exists(valid_data_loc):
        """already saved"""
        return

    encoder = TextEncoder(get_tokenizer())
    map_func = encoder.get_longformer_qa_features
    train_dataset, valid_dataset = _get_processed_dataset(map_func, dataset_name, version_name)

    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    torch.save(train_dataset, data_loc(dataset_name, version_name, train_split_name))
    torch.save(valid_dataset, valid_data_loc)


def _get_processed_dataset(map_function, dataset_name, version_name):
    train_dataset = load_unprocessed_dataset(dataset_name, version_name, nlp.Split.TRAIN)
    valid_dataset = load_unprocessed_dataset(dataset_name, version_name, nlp.Split.VALIDATION)

    print("mapping dataset")
    batch_size = 3000
    train_dataset = train_dataset.map(map_function, load_from_cache_file=False, batch_size=batch_size)
    valid_dataset = valid_dataset.map(map_function, load_from_cache_file=False, batch_size=batch_size)
    return train_dataset, valid_dataset