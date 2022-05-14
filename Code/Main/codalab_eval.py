import argparse
import json
import os
import random
import subprocess
import sys
from os.path import join

import torch
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(join(dir_path_1, 'Code'))
sys.path.append(join(dir_path_1, 'Code.Main'))
sys.path.append(join(dir_path_1, 'Config'))
sys.path.append(join(dir_path_1, 'Checkpoint'))

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_packages():
    # subprocess.call(['pip', 'install', '--upgrade'] + ["torch", "pytorch"])
    try:
        import transformers
    except ModuleNotFoundError:
        print("installing transformers")
        install("transformers")
    try:
        import nlp
    except ModuleNotFoundError:
        print("installing nlp")
        install("nlp")

install_packages()

from Code.Embedding.bert_embedder import TooManyTokens
from Code.Embedding.glove_embedder import NoWordsException
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow

from Config.config import get_config
from Code.Utils.dataset_utils import load_unprocessed_dataset
from Code.Training.wikipoint import Wikipoint
from Checkpoint.checkpoint_utils import model_config_path, load_json_data
from Code.Utils.graph_utils import create_graph

MODEL_CHECKPOINT_NAME = "base_2_trans_bert_freeze_sdp_graph_no_gate_0"


def get_checkpoint_model():
    conf = get_config()
    conf.use_wandb = False

    from Code.Utils.model_utils import get_model
    model, optimizer, scheduler = get_model(MODEL_CHECKPOINT_NAME)
    cfg = load_json_data(model_config_path(model.name))
    conf.set_values(cfg)
    conf.use_wandb = False

    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('first_arg', nargs=1)
    parser.add_argument('second_arg', nargs=1)

    args = parser.parse_args()
    return args


def write_data_to_json():
    import nlp
    split = nlp.Split.VALIDATION
    wiki = list(load_unprocessed_dataset("qangaroo", "wikihop", split))
    with open("wiki_dev.json", "w") as outfile:
        json.dump(wiki, outfile, indent=1)


def get_wiki_graphs(json_path, tokeniser, max_examples=None):
    with open(json_path, "r") as file:
        wiki_examples = json.load(file)
    if max_examples is not None:
        wiki_examples = list(wiki_examples)[:max_examples]
    processed_examples = [Wikipoint(ex, tokeniser=tokeniser) for ex in tqdm(wiki_examples)]
    graphs = [create_graph(ex, tokeniser=tokeniser) for ex in tqdm(processed_examples)]
    return graphs


def infer_on_graphs(model, graphs, out_file_path):
    predictions = {}
    num_guesses = 0
    with torch.no_grad():
        for i, graph in tqdm(enumerate(graphs)):
            graph: HDEGraph
            try:
                loss, predicted = model(graph=graph)
                print("pred:", predicted)
                predictions[graph.example.example_id] = predicted
            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                num_guesses += 1
                predictions[graph.example.example_id] = random.choice(graph.example.candidates)
    with open(out_file_path, "w") as outfile:
        json.dump(predictions, outfile, indent=1)


if __name__ == "__main__":
    # write_data_to_json()

    model = get_checkpoint_model()
    print("loaded model", model)
    args = get_args()
    print("input data json:", args.first_arg, "output_predictions:", args.second_arg)

    graphs = get_wiki_graphs(args.first_arg[0], model.embedder.tokenizer, max_examples=None)
    infer_on_graphs(model, graphs, args.second_arg[0])
