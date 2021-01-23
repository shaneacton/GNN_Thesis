import os
import sys
import time

from tqdm import tqdm

from Code.Config import sysconf
from Code.HDE.hde_glove import HDEGloveEmbed

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

import nlp
from numpy import mean
from torch import optim

from Code.HDE.hde_long_embed import HDELongEmbed
from Code.Training import device
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset

NUM_EPOCHS = 2
PRINT_LOSS_EVERY = 50

print("loading data")

wikihop = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.TRAIN)

hde = HDEGloveEmbed().to(device)
optimizer = optim.SGD(hde.parameters(), lr=0.001, momentum=0.9)

losses = []
last_print = time.time()
print("num examples:", len(wikihop))
for epoch in range(NUM_EPOCHS):
    for i, example in tqdm(enumerate(wikihop)):
        optimizer.zero_grad()
        # print(example)
        answer = example["answer"]
        candidates = example["candidates"]
        query = example["query"]
        supports = example["supports"]

        loss, ans = hde(supports, query, candidates, answer=answer)
        t = time.time()
        loss.backward()
        if sysconf.print_times:
            print("back time:", (time.time() - t))
        t = time.time()
        optimizer.step()
        losses.append(loss.item())

        if len(losses) % PRINT_LOSS_EVERY == 0:
            print("e", epoch, "i", i, "loss:", mean(losses[-PRINT_LOSS_EVERY:-1]), "mean:", mean(losses), "time:", (time.time() - last_print))
            last_print = time.time()
