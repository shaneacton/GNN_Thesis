import os
import pathlib
import sys
import time
from os.path import join, exists

import torch
from tqdm import tqdm


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

import nlp
from numpy import mean
from torch import optim

from Code.Training.Utils.eval_utils import get_acc_and_f1
from Code.Config import sysconf
from Code.HDE.hde_glove import HDEGloveEmbed
from Code.HDE.hde_long_embed import HDELongEmbed
from Code.Training import device
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset

NUM_EPOCHS = 2
PRINT_LOSS_EVERY = 500
MAX_EXAMPLES = 39000

CHECKPOINT_EVERY = 1000
file_path = pathlib.Path(__file__).parent.absolute()
MODEL_SAVE_PATH = join(file_path, "Checkpoint", "hde_model")

print("loading data")

train = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.TRAIN)
test = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.VALIDATION)


def get_model():
    hde = None
    if exists(MODEL_SAVE_PATH):
        try:
            hde = torch.load(MODEL_SAVE_PATH).to(device)
            print("loading checkpoint model")
        except Exception as e:
            print(e)
            print("cannot load model at", MODEL_SAVE_PATH)
    if hde is None:
        hde = HDEGloveEmbed().to(device)

    return hde

hde = get_model()
optimizer = optim.SGD(hde.parameters(), lr=0.001, momentum=0.9)

losses = []
last_print = time.time()
print("num examples:", len(train))
for epoch in range(NUM_EPOCHS):
    answers = []
    predictions = []
    for i, example in tqdm(enumerate(train)):
        optimizer.zero_grad()
        if i >= MAX_EXAMPLES and i != -1:
            break

        if hde.last_example != -1 and i < hde.last_example:  # fast forward
            continue

        # print(example)
        answer = example["answer"]
        candidates = example["candidates"]
        query = example["query"]
        supports = example["supports"]

        loss, predicted = hde(supports, query, candidates, answer=answer)

        answers.append([answer])
        predictions.append(predicted)

        t = time.time()
        loss.backward()
        if sysconf.print_times:
            print("back time:", (time.time() - t))
        t = time.time()
        optimizer.step()
        losses.append(loss.item())

        if len(losses) % PRINT_LOSS_EVERY == 0:
            acc = get_acc_and_f1(answers[-PRINT_LOSS_EVERY:-1], predictions[-PRINT_LOSS_EVERY:-1])
            print("e", epoch, "i", i, "loss:", mean(losses[-PRINT_LOSS_EVERY:-1]), "mean:", mean(losses), "time:", (time.time() - last_print), "acc:", acc)
            last_print = time.time()

        if len(losses) % CHECKPOINT_EVERY == 0:
            print("saving model at e", epoch, "i:", i)
            torch.save(hde, MODEL_SAVE_PATH)

    hde.last_example = -1

    print("e", epoch, "completed. Training acc:", get_acc_and_f1(answers, predictions))
