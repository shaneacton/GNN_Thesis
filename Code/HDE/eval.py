import nlp
from numpy import mean
from tqdm import tqdm

from Code.HDE.Glove.glove_embedder import NoWordsException
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset
from Code.Training.Utils.eval_utils import get_acc_and_f1

test = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.VALIDATION)
MAX_EXAMPLES = -1


def evaluate(hde):
    answers = []
    predictions = []
    chances = []

    hde.eval()
    for i, example in tqdm(enumerate(test)):
        if i >= MAX_EXAMPLES != -1:
            break

        answer = example["answer"]
        candidates = example["candidates"]
        query = example["query"]
        supports = example["supports"]

        try:
            _, predicted = hde(supports, query, candidates, answer=answer)
        except NoWordsException as ne:
            continue

        answers.append([answer])
        predictions.append(predicted)
        chances.append(1./len(candidates))

    hde.last_example = -1

    valid_acc = get_acc_and_f1(answers, predictions)['exact_match']
    print("eval completed. Validation acc:", valid_acc, "chance:", mean(chances))
    return valid_acc
