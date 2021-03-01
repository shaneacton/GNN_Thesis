import random
import time
from statistics import mean

import torch
from tqdm import tqdm

from Code.Embedding.Glove.glove_embedder import NoWordsException
from Code.Embedding.bert_embedder import TooManyTokens
from Code.Training.eval import evaluate
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow
from Code.Training.Utils.training_utils import plot_training_data, save_data, get_model, get_training_results
from Config.config import conf
from Data.dataset_utils import get_processed_wikihop
from Code.Training.Utils.eval_utils import get_acc_and_f1


def train_model(save_path):
    model, optimizer, scheduler = get_model(save_path)
    results = get_training_results(save_path)

    train = get_processed_wikihop(model)
    num_examples = len(train)
    print("num training ex:", num_examples)

    last_print = time.time()
    accumulated_edges = 0

    for epoch in range(conf.num_epochs):
        if model.last_epoch != -1 and epoch < model.last_epoch:  # fast forward
            continue
        random.seed(epoch)
        random.shuffle(train)

        answers = []
        predictions = []
        chances = []
        losses = []
        model.train()

        for i, example in tqdm(enumerate(train)):
            def e_frac():
                return epoch + i/num_examples

            if i >= conf.max_examples != -1:
                break

            if model.last_example != -1 and i < model.last_example:  # fast forward
                continue

            try:
                graph = model.create_graph(example)
                num_edges = len(graph.unique_edges)
                if accumulated_edges + num_edges > conf.max_accumulated_edges:  # always true if mae=-1
                    """
                        this new graph would send us over the accumulated edges budget,
                        so we must first wipe previous gradients by stepping
                    """
                    optimizer.step()
                    if conf.use_lr_scheduler:
                        scheduler.step(epoch=e_frac())
                    optimizer.zero_grad()
                    accumulated_edges = 0

                loss, predicted = model(example, graph=graph)
                loss.backward()
                accumulated_edges += num_edges

            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                continue

            answers.append([example.answer])
            predictions.append(predicted)
            chances.append(1. / len(example.candidates))

            t = time.time()
            if conf.print_times:
                print("back time:", (time.time() - t))

            losses.append(loss.item())

            if len(losses) % conf.print_loss_every == 0:  # print loss
                acc = get_acc_and_f1(answers[-conf.print_loss_every:-1], predictions[-conf.print_loss_every:-1])['exact_match']
                mean_loss = mean(losses[-conf.print_loss_every:-1])
                print("e", epoch, "i", i, "loss:", mean_loss, "mean:", mean(losses),
                      "time:", (time.time() - last_print), "acc:", acc, "chance:", mean(chances[-conf.print_loss_every:-1]))
                last_print = time.time()

                results["losses"].append(mean_loss)
                results["train_accs"].append(acc)

            if len(losses) % conf.checkpoint_every == 0:  # save model and data
                print("saving model at e", epoch, "i:", i)
                model.last_example = i
                model.last_epoch = epoch
                model_save_data = {"model": model, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()}
                torch.save(model_save_data, save_path)
                plot_training_data(results, save_path, conf.print_loss_every, num_examples)
                save_data(results, save_path)
        model.last_example = -1

        print("e", epoch, "completed. Training acc:", get_acc_and_f1(answers, predictions)['exact_match'],
              "chance:", mean(chances) if len(chances) > 0 else 0)

        valid_acc = evaluate(model)
        results["valid_accs"].append(valid_acc)

        plot_training_data(results, save_path, conf.print_loss_every, num_examples)