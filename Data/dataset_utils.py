import pathlib
import pickle
from os.path import exists, join

import nlp
from tqdm import tqdm

from Code.HDE.wikipoint import Wikipoint
from Code.Training.Utils.training_utils import save_data
from Config.config import conf

DATA_FOLDER = str(pathlib.Path(__file__).parent.absolute())
print("Data folder:", DATA_FOLDER)


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


def get_processed_wikihop(model, split=nlp.Split.TRAIN):
    global has_loaded

    file_name = model.embedder_name + "_" + split._name + ".data"
    data_path = join(DATA_FOLDER, file_name)

    if exists(data_path):  # has been processed before
        print("loading preprocessed wikihop", split)
        filehandler = open(data_path, 'rb')
        data = pickle.load(filehandler)
        filehandler.close()
        return data

    print("loading wikihop unprocessed")
    data = list(load_unprocessed_dataset("qangaroo", "wikihop", split))
    data = data[:conf.max_examples] if conf.max_examples > 0 else data
    print("num examples:", len(data))

    print("processing wikihop", split)
    if model.embedder_name == "bert":
        processed_examples = [Wikipoint(ex, tokeniser=model.embedder.tokenizer) for ex in tqdm(data)]
    else:
        processed_examples = [Wikipoint(ex, glove_embedder=model.embedder) for ex in tqdm(data)]
    save_data(processed_examples, DATA_FOLDER + "/", suffix=file_name)
    return processed_examples