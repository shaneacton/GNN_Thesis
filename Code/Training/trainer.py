import time
from statistics import mean

import torch

from Code.Embedding.Glove.glove_embedder import NoWordsException
from Code.Embedding.bert_embedder import TooManyTokens
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow
from Code.Training.Utils.eval_utils import get_acc_and_f1
from Code.Training.Utils.training_utils import plot_training_data, save_data, get_model, get_training_results
from Code.Training.eval import evaluate
from Code.Training.graph_gen import GraphGenerator, SKIP
from Config.config import conf
from Data.dataset_utils import get_processed_wikihop


def train_model(save_path):
    model, optimizer, scheduler = get_model(save_path)
    results = get_training_results(save_path)

    train_gen = GraphGenerator(get_processed_wikihop(model), model=model)

    accumulated_edges = 0
    for epoch in range(conf.num_epochs):
        if model.last_epoch != -1 and epoch < model.last_epoch:  # fast forward
            continue
        train_gen.shuffle(epoch)

        answers = []
        predictions = []
        chances = []
        losses = []
        model.train()

        start_time = time.time()
        for i, graph in enumerate(train_gen.graphs(start_at=model.last_example)):
            def e_frac():
                return epoch + i/train_gen.num_examples

            if i >= conf.max_examples != -1:
                break

            if graph == SKIP:
                continue

            try:
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

                loss, predicted = model(graph=graph)
                t = time.time()
                loss.backward()
                if conf.print_times:
                    print("back time:", (time.time() - t))

                accumulated_edges += num_edges

            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                continue

            answers.append([graph.example.answer])
            predictions.append(predicted)
            chances.append(1. / len(graph.example.candidates))

            losses.append(loss.item())

            if len(losses) % conf.print_loss_every == 0:  # print loss
                acc = get_acc_and_f1(answers[-conf.print_loss_every:-1], predictions[-conf.print_loss_every:-1])['exact_match']
                mean_loss = mean(losses[-conf.print_loss_every:-1])
                print("e", epoch, "i", i, "loss:", mean_loss, "acc:", acc, "chance:", mean(chances[-conf.print_loss_every:-1]))

                results["losses"].append(mean_loss)
                results["train_accs"].append(acc)

            if len(losses) % conf.checkpoint_every == 0:  # save model and data
                save_time = time.time()
                print("saving model at e", epoch, "i:", i)
                model.last_example = i
                model.last_epoch = epoch
                model_save_data = {"model": model, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()}
                torch.save(model_save_data, save_path)
                plot_training_data(results, save_path, conf.print_loss_every, train_gen.num_examples)
                save_data(results, save_path)
                save_time = time.time() - save_time
                start_time += save_time
        model.last_example = -1

        print("e", epoch, "completed. Training acc:", get_acc_and_f1(answers, predictions)['exact_match'],
              "chance:", mean(chances) if len(chances) > 0 else 0, "time:", (time.time() - start_time))


        valid_acc = evaluate(model)
        results["valid_accs"].append(valid_acc)

        plot_training_data(results, save_path, conf.print_loss_every, train_gen.num_examples)