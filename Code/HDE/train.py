import nlp
from numpy import mean
from torch import optim

from Code.HDE.hde import HDE
from Code.Training import device
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset

NUM_EPOCHS = 2
PRINT_LOSS_EVERY = 50

print("loading data")

wikihop = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.TRAIN)

hde = HDE().to(device)
optimizer = optim.SGD(hde.parameters(), lr=0.001, momentum=0.9)

losses = []
for epoch in range(NUM_EPOCHS):
    for i, example in enumerate(wikihop):
        optimizer.zero_grad()
        # print(example)
        answer = example["answer"]
        candidates = example["candidates"]
        query = example["query"]
        supports = example["supports"]

        loss, ans = hde(supports, query, candidates, answer=answer)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if len(losses) % PRINT_LOSS_EVERY == 0:
            print("e", epoch, "i", i, "loss:", mean(losses[-PRINT_LOSS_EVERY:-1]))
