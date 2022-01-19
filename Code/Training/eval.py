import time

import nlp
import torch
from numpy import mean

from Checkpoint.checkpoint_utils import load_status, save_status
from Code.Embedding.glove_embedder import NoWordsException
from Code.Embedding.bert_embedder import TooManyTokens
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow
from Code.Utils.dataset_utils import get_wikihop_graphs
from Code.Utils.eval_utils import get_acc_and_f1
from Config.config import conf


def evaluate(hde, program_start_time=None):
    print("starting evaluation")
    graphs = get_wikihop_graphs(nlp.Split.VALIDATION, hde.embedder)

    answers = []
    predictions = []
    chances = []

    hde.eval()

    with torch.no_grad():
        for i, graph in enumerate(graphs):
            if i >= conf.max_examples != -1:
                break
            if time.time() - program_start_time > conf.max_runtime_seconds != -1:
                print("reached max run time. shutting down so the program can exit safely")
                status = load_status(conf.model_name)
                status["running"] = False
                save_status(conf.model_name, status)
                exit()
            try:
                _, predicted = hde(graph)
            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                continue

            answers.append([graph.example.answer])
            predictions.append(predicted)
            chances.append(1./len(graph.example.candidates))

    hde.last_example = -1

    valid_acc = get_acc_and_f1(answers, predictions)['exact_match']
    print("eval completed. Validation acc:", valid_acc, "chance:", mean(chances))
    return valid_acc
