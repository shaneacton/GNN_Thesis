import os
import pathlib
import pickle
import random
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

from Viz.loss_visualiser import plot_training_data
from Code.HDE.hde_glove_stack import HDEGloveStack
from Code.HDE.eval import evaluate
from Code.HDE.Glove.glove_embedder import NoWordsException
from Code.Training.Utils.eval_utils import get_acc_and_f1
from Code.Config import sysconf, gcc
from Code.Training import device
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset

NUM_EPOCHS = 8
PRINT_LOSS_EVERY = 500
MAX_EXAMPLES = -1

CHECKPOINT_EVERY = 43000
file_path = pathlib.Path(__file__).parent.absolute()
CHECKPOINT_FOLDER = join(file_path, "Checkpoint")

# MODEL_NAME = "hde_model_stack_large_deep"
MODEL_NAME = "hde_model_stack_large"

MODEL_SAVE_PATH = join(CHECKPOINT_FOLDER, MODEL_NAME)

print("loading data")

train = load_unprocessed_dataset("qangaroo", "wikihop", nlp.Split.TRAIN)

plot_losses = []
plot_train_accs = []
plot_valid_accs = []


def get_model():
    global plot_losses
    global plot_train_accs
    global plot_valid_accs

    hde = None
    if exists(MODEL_SAVE_PATH):
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH)
            hde = checkpoint["model"].to(device)
            optimizer = optim.SGD(hde.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("loading checkpoint model at:", MODEL_SAVE_PATH, "with",
                  sum(p.numel() for p in hde.parameters() if p.requires_grad), "trainable params")
            filehandler = open(MODEL_SAVE_PATH + ".data", 'rb')
            train_data = pickle.load(filehandler)
            plot_losses, plot_train_accs, plot_valid_accs = train_data["losses"], train_data["train_accs"], \
                                                            train_data["valid_accs"]
            print("loaded losses:", len(plot_losses), plot_losses)
        except Exception as e:
            print(e)
            print("cannot load model at", MODEL_SAVE_PATH)
    if hde is None:
        hde = HDEGloveStack(hidden_size=200, embedded_dims=100, num_layers=2).to(device)
        optimizer = optim.SGD(hde.parameters(), lr=0.001)
        print("inited model", repr(hde), "with:", sum(p.numel() for p in hde.parameters() if p.requires_grad), "trainable params")

    return hde, optimizer


hde, optimizer = get_model()


last_print = time.time()
num_examples = len(train)
print("num examples:", num_examples)


def plot_training_results(losses, train_accs, valid_accs):
    path = MODEL_SAVE_PATH + "_losses.png"
    num_prints = len(losses)
    num_trained_examples = num_prints * PRINT_LOSS_EVERY
    num_epochs = num_trained_examples / num_examples
    epochs = [num_epochs * i/len(losses) for i in range(len(losses))]
    plot_training_data(losses, accuracies=train_accs, show=False, save_path=path, epochs=epochs, valid_accs=valid_accs)
    filehandler = open(MODEL_SAVE_PATH + ".data", 'wb')
    data = {"losses": losses, "train_accs": train_accs, "valid_accs": valid_accs}
    pickle.dump(data, filehandler)


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
            # loss, predicted = hde(supports, query, candidates, answer=answer)
            loss, predicted = torch.tensor(0), random.choice(candidates)
        except NoWordsException as ne:
            continue

        answers.append([answer])
        predictions.append(predicted)
        chances.append(1./len(candidates))

        t = time.time()
        # loss.backward()
        if sysconf.print_times:
            print("back time:", (time.time() - t))
        t = time.time()
        # optimizer.step()
        losses.append(loss.item())

        if len(losses) % PRINT_LOSS_EVERY == 0:  # print loss
            acc = get_acc_and_f1(answers[-PRINT_LOSS_EVERY:-1], predictions[-PRINT_LOSS_EVERY:-1])['exact_match']
            mean_loss = mean(losses[-PRINT_LOSS_EVERY:-1])
            print("e", epoch, "i", i, "loss:", mean_loss, "mean:", mean(losses),
                  "time:", (time.time() - last_print), "acc:", acc, "chance:", mean(chances[-PRINT_LOSS_EVERY:-1]))
            last_print = time.time()

            plot_losses.append(mean_loss)
            plot_train_accs.append(acc)

        if len(losses) % CHECKPOINT_EVERY == 0:  # save model
            print("saving model at e", epoch, "i:", i)
            hde.last_example = i
            hde.last_epoch = epoch
            torch.save({"model": hde, "optimizer_state_dict": optimizer.state_dict()}, MODEL_SAVE_PATH)
            plot_training_results(plot_losses, plot_train_accs, plot_valid_accs)

    hde.last_example = -1

    print("e", epoch, "completed. Training acc:", get_acc_and_f1(answers, predictions)['exact_match'],
          "chance:", mean(chances))

    valid_acc = evaluate(hde)
    plot_valid_accs.append(valid_acc)

plot_training_results(plot_losses, plot_train_accs, plot_valid_accs)