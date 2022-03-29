from __future__ import annotations

import pickle
from os.path import exists, join
from typing import TYPE_CHECKING, List

import nlp
from tqdm import tqdm

from Code.Training.wikipoint import Wikipoint
from Checkpoint.checkpoint_utils import save_binary_data
from Code.Utils.graph_utils import create_graph
from Config.config import get_config
from Data import DATA_FOLDER

if TYPE_CHECKING:
    from Code.HDE.hde_model import HDEModel
    from Code.HDE.Graph.graph import HDEGraph


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


def get_wikihop_graphs(split=nlp.Split.TRAIN, embedder=None) -> List[HDEGraph]:
    emb_name = get_config().embedder_type + "(" + repr(get_config().embedded_dims) + ")"
    if get_config().embedder_type != "glove":
        emb_name += "_" + get_config().bert_name.replace("\\", "_").replace("/", "_")
    file_name = emb_name + "_" + split._name
    ent_name = "detected" if not get_config().use_special_entities else ("special" if not get_config().use_detected_entities else "det&spec")
    file_name += "_" + ent_name
    if get_config().use_codocument_edges:
        file_name += "_codoc"

    if get_config().use_compliment_edges:
        file_name += "_comp"

    if get_config().use_sentence_nodes:
        if get_config().use_all_sentences:
            file_name += "_allSents"
        else:
            file_name += "_sents"
        if get_config().connect_sent2sent:
            file_name += "Seq"

    if get_config().bidirectional_edge_types:
        file_name += "_bi"

    file_name = get_config().dataset + "_" + file_name + ".data"

    if get_config().run_args.processed_data_path:
        data_path = join(get_config().run_args.processed_data_path, file_name)
    else:
        data_path = join(DATA_FOLDER, file_name)
    print("processed data path:", data_path)

    if exists(data_path):  # has been processed before
        print("loading preprocessed wikihop", split)
        filehandler = open(data_path, 'rb')
        graphs = pickle.load(filehandler)
        print("loaded", len(graphs), "graphs")
        filehandler.close()
        return graphs

    print("loading",get_config().dataset, "unprocessed")
    data = list(load_unprocessed_dataset("qangaroo", get_config().dataset, split))
    data = data[:get_config().max_examples] if get_config().max_examples > 0 else data
    print("num examples:", len(data))

    print("processing", get_config().dataset, split)
    if get_config().embedder_type == "bert":
        print("tokenising text for bert")
        processed_examples = [Wikipoint(ex, tokeniser=embedder.tokenizer) for ex in tqdm(data)]
        print("creating graphs")
        graphs = [create_graph(ex, tokeniser=embedder.tokenizer) for ex in tqdm(processed_examples)]
    else:
        print("tokenising text for glove")
        processed_examples = [Wikipoint(ex, glove_embedder=embedder) for ex in tqdm(data)]
        print("creating graphs")
        graphs = [create_graph(ex, glove_embedder=embedder) for ex in tqdm(processed_examples)]
    print("created graphs with edge types:", graphs[0].unique_edge_types)

    print("saving", len(graphs), "graphs")
    save_binary_data(graphs, data_path)
    return graphs