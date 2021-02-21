import nlp
import torch
from numpy import mean
from tqdm import tqdm

from Code.HDE.Glove.glove_embedder import NoWordsException
from Code.HDE.hde_glove import PadVolumeOverflow, TooManyEdges
from Code.HDE.training_utils import get_processed_wikihop
from Code.Training.Utils.eval_utils import get_acc_and_f1

_test = None


def get_test(save_path, embedder, max_examples=-1):
    global _test
    if _test is None:
        _test = get_processed_wikihop(save_path, embedder, max_examples=max_examples, split=nlp.Split.VALIDATION)
    return _test


def evaluate(hde, save_path, max_examples):
    test = get_test(save_path, hde.embedder, max_examples)

    answers = []
    predictions = []
    chances = []

    hde.eval()

    with torch.no_grad():
        for i, example in tqdm(enumerate(test)):
            if i >= max_examples != -1:
                break
            try:
                _, predicted = hde(example)
            except (NoWordsException, PadVolumeOverflow, TooManyEdges) as ne:
                continue

            answers.append([example.answer])
            predictions.append(predicted)
            chances.append(1./len(example.candidates))

    hde.last_example = -1

    valid_acc = get_acc_and_f1(answers, predictions)['exact_match']
    print("eval completed. Validation acc:", valid_acc, "chance:", mean(chances))
    return valid_acc
