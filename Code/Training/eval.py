import nlp
import torch
from numpy import mean
from tqdm import tqdm

from Code.Embedding.Glove.glove_embedder import NoWordsException
from Code.Embedding.bert_embedder import TooManyTokens
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow
from Config.config import conf
from Data.dataset_utils import get_processed_wikihop
from Code.Training.Utils.eval_utils import get_acc_and_f1

_test = None


def get_test(model):
    global _test
    if _test is None:
        _test = get_processed_wikihop(model, split=nlp.Split.VALIDATION)
        print("num valid ex:", len(_test))
    return _test


def evaluate(hde):
    test = get_test(hde)

    answers = []
    predictions = []
    chances = []

    hde.eval()

    with torch.no_grad():
        for i, example in tqdm(enumerate(test)):
            if i >= conf.max_examples != -1:
                break
            try:
                _, predicted = hde(example)
            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                continue

            answers.append([example.answer])
            predictions.append(predicted)
            chances.append(1./len(example.candidates))

    hde.last_example = -1

    valid_acc = get_acc_and_f1(answers, predictions)['exact_match']
    print("eval completed. Validation acc:", valid_acc, "chance:", mean(chances))
    return valid_acc
