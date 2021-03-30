import time
from math import floor
from statistics import mean

from nlp import tqdm

from Checkpoint.checkpoint_utils import save_model, set_status_value, duplicate_checkpoint_folder
from Code.Embedding.Glove.glove_embedder import NoWordsException
from Code.Embedding.bert_embedder import TooManyTokens
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow
from Code.Training import set_gpu
from Code.Training.Utils.eval_utils import get_acc_and_f1
from Code.Training.Utils.model_utils import get_model
from Code.Training.Utils.training_utils import plot_training_data, save_training_results, get_training_results
from Code.Training.eval import evaluate
from Code.Training.graph_gen import GraphGenerator, SKIP
from Config.config import conf
from Data.dataset_utils import get_processed_wikihop
from Viz.wandb_utils import use_wandb, wandb_run


def train_model(name, gpu_num=0):
    set_gpu(gpu_num)
    model, optimizer, scheduler = get_model(name)
    if use_wandb:
        try:
            wandb_run().watch(model)
        except:
            pass
    results = get_training_results(name)

    train_gen = GraphGenerator(get_processed_wikihop(model), model=model)

    accumulated_edges = 0

    start_time = time.time()

    for epoch in range(conf.num_epochs):
        if model.last_epoch != -1 and epoch < model.last_epoch:  # fast forward
            continue
        train_gen.shuffle(epoch)

        answers = []
        predictions = []
        chances = []
        losses = []
        model.train()

        epoch_start_time = time.time()
        num_discarded = 0
        for i, graph in tqdm(enumerate(train_gen.graphs(start_at=model.last_example))):
            def e_frac():
                return epoch + i/train_gen.num_examples

            if i >= conf.max_examples != -1:
                break

            if graph == SKIP:
                continue

            if time.time() - start_time > conf.max_runtime_seconds != -1:
                print("reached max run time. shutting down so the program can exit safely")
                exit()

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
                    if conf.show_memory_usage_data:
                        print("accumulated edges (", accumulated_edges, ") is over max. stepping optim:")
                    accumulated_edges = 0

                loss, predicted = model(graph=graph)
                t = time.time()
                loss.backward()
                if conf.print_times:
                    print("back time:", (time.time() - t))

                accumulated_edges += num_edges
                if conf.show_memory_usage_data:
                    print("accumulated edges:", accumulated_edges)

            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                num_discarded += 1
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
                if use_wandb:
                    wandb_run().log({"loss": mean_loss, "train_acc": acc, "epoch": e_frac()})

            if len(losses) % conf.checkpoint_every == 0:  # save model and data
                epoch_start_time = save_training_states(e_frac(), epoch_start_time, i, model, name, optimizer, results,
                                                        scheduler, start_time, train_gen.num_examples)
        model.last_example = -1

        print("e", epoch, "completed. Training acc:", get_acc_and_f1(answers, predictions)['exact_match'],
              "chance:", mean(chances) if len(chances) > 0 else 0, "time:", (time.time() - epoch_start_time),
              "num discarded:", num_discarded)

        valid_acc = evaluate(model)
        set_status_value(name, "completed_epochs", epoch)

        if use_wandb:
            wandb_run().log({"valid_acc": valid_acc, "epoch": epoch + 1})
        results["valid_accs"].append(valid_acc)

        plot_training_data(results, name, conf.print_loss_every, train_gen.num_examples)

    set_status_value(name, "finished", True)


def save_training_states(epoch, epoch_start_time, i, model, name, optimizer, results, scheduler, start_time, num_examples):
    if time.time() - start_time + 5 * 60 > conf.max_runtime_seconds != -1:
        # end 5m early before saving, as this can take long enough to send us over the max runtime
        print("reached max run time. shutting down so the program can exit safely")
        exit()
    save_time = time.time()
    print("saving model at e", epoch, "i:", i)
    model.last_example = i
    model.last_epoch = floor(epoch)
    save_model(model, optimizer, scheduler)
    plot_training_data(results, name, conf.print_loss_every, num_examples)
    save_training_results(results, name)
    save_time = time.time() - save_time
    epoch_start_time += save_time
    set_status_value(name, "completed_epochs", epoch)
    """upon successful saving. make a backup so that if the next save fails, we can continue from here next time"""
    duplicate_checkpoint_folder(name)

    return epoch_start_time
