import random
import time
from statistics import mean

import torch
from tqdm import tqdm

from Code.Config import sysconf
from Code.HDE.Glove.glove_embedder import NoWordsException
from Code.HDE.eval import evaluate
from Code.HDE.hde_glove import PadVolumeOverflow, TooManyEdges
from Code.HDE.training_utils import plot_training_data, save_training_data, get_model, get_training_data, save_config, \
    get_processed_wikihop
from Code.Training.Utils.eval_utils import get_acc_and_f1


def train_model(save_path, num_epochs=5, max_examples=-1, print_loss_every=500, checkpoint_every=1000, **model_cfg):
    model, optimizer = get_model(save_path, **model_cfg)
    results = get_training_data(save_path)
    save_config(model_cfg, save_path)

    train = get_processed_wikihop(save_path, model.embedder, max_examples=max_examples)
    num_examples = len(train)

    last_print = time.time()
    for epoch in range(num_epochs):
        if model.last_epoch != -1 and epoch < model.last_epoch:  # fast forward
            continue
        random.seed(epoch)
        # random.shuffle(train)

        answers = []
        predictions = []
        chances = []
        losses = []
        model.train()

        for i, example in tqdm(enumerate(train)):
            optimizer.zero_grad()
            if i >= max_examples != -1:
                break

            if model.last_example != -1 and i < model.last_example:  # fast forward
                continue

            try:
                loss, predicted = model(example)
            except (NoWordsException, PadVolumeOverflow, TooManyEdges) as ne:
                continue

            answers.append([example.answer])
            predictions.append(predicted)
            chances.append(1. / len(example.candidates))

            t = time.time()
            loss.backward()
            if sysconf.print_times:
                print("back time:", (time.time() - t))
            optimizer.step()
            losses.append(loss.item())

            if len(losses) % print_loss_every == 0:  # print loss
                acc = get_acc_and_f1(answers[-print_loss_every:-1], predictions[-print_loss_every:-1])['exact_match']
                mean_loss = mean(losses[-print_loss_every:-1])
                print("e", epoch, "i", i, "loss:", mean_loss, "mean:", mean(losses),
                      "time:", (time.time() - last_print), "acc:", acc, "chance:", mean(chances[-print_loss_every:-1]))
                last_print = time.time()

                results["losses"].append(mean_loss)
                results["train_accs"].append(acc)

            if len(losses) % checkpoint_every == 0:  # save model and data
                print("saving model at e", epoch, "i:", i)
                model.last_example = i
                model.last_epoch = epoch
                torch.save({"model": model, "optimizer_state_dict": optimizer.state_dict()}, save_path)
                plot_training_data(results, save_path, print_loss_every, num_examples)
                save_training_data(results, save_path)
        model.last_example = -1

        print("e", epoch, "completed. Training acc:", get_acc_and_f1(answers, predictions)['exact_match'],
              "chance:", mean(chances))

        valid_acc = evaluate(model, save_path, max_examples)
        results["valid_accs"].append(valid_acc)

        plot_training_data(results, save_path, print_loss_every, num_examples)