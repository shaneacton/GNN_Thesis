import nlp

from Code.Training.Utils.dataset_utils import load_unprocessed_dataset

wikihop = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.TRAIN)

for example in wikihop:
    print(example)
    answer = example["answer"]
    candidates = example["candidates"]
    query = example["query"]
    supports = example["supports"]

    