import os
import pathlib
import random
import sys
import time
from os.path import join

import torch
from tqdm import tqdm

from Code.HDE.training_utils import get_model, get_training_data, plot_training_data, save_training_data

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

import nlp
from numpy import mean

from Code.HDE.eval import evaluate
from Code.HDE.Glove.glove_embedder import NoWordsException
from Code.Training.Utils.eval_utils import get_acc_and_f1
from Code.Config import sysconf, gcc
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset

NUM_EPOCHS = 5
PRINT_LOSS_EVERY = 500
MAX_EXAMPLES = -1

CHECKPOINT_EVERY = 1000
file_path = pathlib.Path(__file__).parent.absolute()
CHECKPOINT_FOLDER = join(file_path, "Checkpoint")

# MODEL_NAME = "hde_model_stack_large_deep"
MODEL_NAME = "hde_model_stack_large"

MODEL_SAVE_PATH = join(CHECKPOINT_FOLDER, MODEL_NAME)

print("loading data")

train = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.TRAIN)


hde, optimizer = get_model(MODEL_SAVE_PATH, num_layers=4)
results = get_training_data(MODEL_SAVE_PATH)


last_print = time.time()
num_examples = len(train)
print("num examples:", num_examples)


for epoch in range(NUM_EPOCHS):
    if hde.last_epoch != -1 and epoch < hde.last_epoch:  # fast forward
        continue

    answers = []
    predictions = []
    chances = []
    losses = []
    hde.train()

    for i, example in tqdm(enumerate(train)):
        optimizer.zero_grad()
        if i >= MAX_EXAMPLES != -1:
            break

        if hde.last_example != -1 and i < hde.last_example:  # fast forward
            continue

        # print(example)
        answer = example["answer"]
        candidates = example["candidates"]
        query = example["query"]
        supports = example["supports"]
        supports = [s[:gcc.max_context_chars] if gcc.max_context_chars != -1 else s for s in supports]

        try:
            loss, predicted = hde(supports, query, candidates, answer=answer)
            # loss, predicted = torch.tensor(0), random.choice(candidates)
        except NoWordsException as ne:
            continue

        answers.append([answer])
        predictions.append(predicted)
        chances.append(1./len(candidates))

        t = time.time()
        loss.backward()
        if sysconf.print_times:
            print("back time:", (time.time() - t))
        t = time.time()
        optimizer.step()
        losses.append(loss.item())

        if len(losses) % PRINT_LOSS_EVERY == 0:  # print loss
            acc = get_acc_and_f1(answers[-PRINT_LOSS_EVERY:-1], predictions[-PRINT_LOSS_EVERY:-1])['exact_match']
            mean_loss = mean(losses[-PRINT_LOSS_EVERY:-1])
            print("e", epoch, "i", i, "loss:", mean_loss, "mean:", mean(losses),
                  "time:", (time.time() - last_print), "acc:", acc, "chance:", mean(chances[-PRINT_LOSS_EVERY:-1]))
            last_print = time.time()

            results["losses"].append(mean_loss)
            results["train_accs"].append(acc)

        if len(losses) % CHECKPOINT_EVERY == 0:  # save model and data
            print("saving model at e", epoch, "i:", i)
            hde.last_example = i
            hde.last_epoch = epoch
            torch.save({"model": hde, "optimizer_state_dict": optimizer.state_dict()}, MODEL_SAVE_PATH)
            plot_training_data(results, MODEL_SAVE_PATH, PRINT_LOSS_EVERY, num_examples)
            save_training_data(results, MODEL_SAVE_PATH)
    hde.last_example = -1

    print("e", epoch, "completed. Training acc:", get_acc_and_f1(answers, predictions)['exact_match'],
          "chance:", mean(chances))

    valid_acc = evaluate(hde)
    results["valid_accs"].append(valid_acc)

plot_training_data(results, MODEL_SAVE_PATH, PRINT_LOSS_EVERY, num_examples)