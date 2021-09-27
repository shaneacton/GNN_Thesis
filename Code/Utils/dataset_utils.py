from __future__ import annotations

import pickle
from os.path import exists, join
from typing import TYPE_CHECKING

import nlp
from tqdm import tqdm

from Code.Training.wikipoint import Wikipoint
from Checkpoint.checkpoint_utils import save_binary_data
from Code.Utils.graph_utils import create_graph
from Config.config import conf
from Data import DATA_FOLDER

if TYPE_CHECKING:
    from Code.HDE.hde_model import HDEModel


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


def get_wikihop_graphs(model: HDEModel, split=nlp.Split.TRAIN):
    global has_loaded

    emb_name = conf.embedder_type + "(" + repr(conf.embedded_dims) + ")"
    file_name = emb_name + "_" + split._name + "_special.data"
    if conf.run_args.processed_data_path:
        data_path = join(conf.run_args.processed_data_path, file_name)
    else:
        data_path = join(DATA_FOLDER, file_name)

    if exists(data_path):  # has been processed before
        print("loading preprocessed wikihop", split)
        filehandler = open(data_path, 'rb')
        graphs = pickle.load(filehandler)
        print("loaded", len(graphs), "graphs")
        filehandler.close()
        return graphs

    print("loading wikihop unprocessed")
    data = list(load_unprocessed_dataset("qangaroo", "wikihop", split))
    data = data[:conf.max_examples] if conf.max_examples > 0 else data
    print("num examples:", len(data))

    print("processing wikihop", split)
    if conf.embedder_type == "bert":
        print("tokenising text for bert")
        processed_examples = [Wikipoint(ex, tokeniser=model.embedder.tokenizer) for ex in tqdm(data)]
        print("creating graphs")
        graphs = [create_graph(ex, tokeniser=model.embedder.tokenizer) for ex in tqdm(processed_examples)]
    else:
        print("tokenising text for glove")
        processed_examples = [Wikipoint(ex, glove_embedder=model.embedder) for ex in tqdm(data)]
        print("creating graphs")
        graphs = [create_graph(ex, glove_embedder=model.embedder) for ex in tqdm(processed_examples)]

    print("saving", len(graphs), "graphs")
    save_binary_data(graphs, data_path)
    return graphs