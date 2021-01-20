import nlp
from torch import optim

from Code.HDE.hde import HDE
from Code.Training import device
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset

NUM_EPOCHS = 2

wikihop = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.TRAIN)

hde = HDE().to(device)
optimizer = optim.SGD(hde.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    for example in wikihop:
        print(example)
        answer = example["answer"]
        candidates = example["candidates"]
        query = example["query"]
        supports = example["supports"]

        out = hde(supports, query, candidates)
    