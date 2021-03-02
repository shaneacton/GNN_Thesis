import nlp
import torch
from numpy import mean

from Code.Embedding.Glove.glove_embedder import NoWordsException
from Code.Embedding.bert_embedder import TooManyTokens
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow
from Code.Training.Utils.eval_utils import get_acc_and_f1
from Code.Training.graph_gen import GraphGenerator
from Config.config import conf
from Data.dataset_utils import get_processed_wikihop

_test = None


def get_test(model):
    global _test
    if _test is None:
        _test = get_processed_wikihop(model, split=nlp.Split.VALIDATION)
        print("num valid ex:", len(_test))
    return _test


def evaluate(hde):
    test_gen = GraphGenerator(get_test(hde), hde)

    answers = []
    predictions = []
    chances = []

    hde.eval()

    with torch.no_grad():
        for i, graph in enumerate(test_gen.graphs()):
            if i >= conf.max_examples != -1:
                break
            try:
                _, predicted = hde(graph=graph)
            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                continue

            answers.append([graph.example.answer])
            predictions.append(predicted)
            chances.append(1./len(graph.example.candidates))

    hde.last_example = -1

    valid_acc = get_acc_and_f1(answers, predictions)['exact_match']
    print("eval completed. Validation acc:", valid_acc, "chance:", mean(chances))
    return valid_acc
