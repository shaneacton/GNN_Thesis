import random
import time
from math import floor
from nlp import tqdm

from Checkpoint.checkpoint_utils import save_model, set_status_value, duplicate_checkpoint_folder, load_status, \
    save_status
from Code.Embedding.bert_embedder import TooManyTokens
from Code.Embedding.glove_embedder import NoWordsException
from Code.HDE.hde_model import TooManyEdges, PadVolumeOverflow
from Code.Training import set_gpu
from Code.Training.eval import evaluate
from Code.Training.timer import log_time
from Code.Training.training_results import TrainingResults
from Code.Utils.model_utils import get_model
from Code.Utils.training_utils import get_training_results, save_training_results
from Config.config import conf
from Code.Utils.dataset_utils import get_wikihop_graphs
from Code.Utils.wandb_utils import use_wandb, wandb_run


def train_model(name, gpu_num=0, program_start_time=-1):
    if program_start_time == -1:
        program_start_time = time.time() - 120
    set_gpu(gpu_num)
    print("max edges:", conf.max_edges, "max pad volume:", conf.max_pad_volume)
    model, optimizer, scheduler = get_model(name)
    if use_wandb:
        try:
            wandb_run().watch(model)
        except:
            pass

    graphs = get_wikihop_graphs(model)
    accumulated_edges = 0

    start_time = time.time()

    training_results = get_training_results(name)

    for epoch in range(conf.num_epochs):
        if model.last_epoch != -1 and epoch < model.last_epoch:  # fast forward
            continue
        random.Random(4).shuffle(graphs)
        model.train()

        epoch_start_time = time.time()
        num_discarded = 0
        num_accumulation_steps = 0
        num_fastforward_examples = max(model.last_example, 0)
        epoch_graphs = graphs if model.last_example == -1 else graphs[num_fastforward_examples:]
        for i, graph in tqdm(enumerate(epoch_graphs)):
            def e_frac():
                return epoch + (i+num_fastforward_examples)/len(graphs)

            if i >= conf.max_examples != -1:
                break

            if time.time() - program_start_time > conf.max_runtime_seconds != -1:
                times_up()

            try:
                num_edges = len(graph.unique_edges)
                if accumulated_edges + num_edges > conf.max_accumulated_edges:  # always true if mae=-1
                    """
                        this new graph would send us over the accumulated edges budget,
                        so we must first wipe previous gradients by stepping
                    """
                    num_accumulation_steps = max(1, num_accumulation_steps)
                    if conf.scale_accumulated_gradients:
                        for p in model.parameters():
                            if p.grad is not None and p.requires_grad:
                                p.grad /= num_accumulation_steps

                    optimizer.step()
                    num_accumulation_steps = 0
                    if conf.use_lr_scheduler:
                        scheduler.step(epoch=e_frac())
                    optimizer.zero_grad()
                    if conf.show_memory_usage_data:
                        print("accumulated edges (", accumulated_edges, ") is over max. stepping optim:")
                    accumulated_edges = 0

                loss, predicted = model(graph=graph)
                t = time.time()
                loss.backward()
                num_accumulation_steps += 1

                log_time("Backwards pass", time.time() - t)

                accumulated_edges += num_edges
                if conf.show_memory_usage_data:
                    print("accumulated edges:", accumulated_edges)

            except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
                num_discarded += 1
                continue

            training_results.report_step(loss.item(), predicted, graph.example.answer, len(graph.example.candidates))

            if len(training_results.all_losses) % conf.print_loss_every == 0:  # print loss
                training_results.log_last_steps(e_frac())

            if len(training_results.all_losses) % conf.checkpoint_every == 0:  # save model and data
                # saving takes a few minutes. We should check for an early stoppage to ensure program closes well
                if conf.max_runtime_seconds != -1 and time.time() - program_start_time > conf.max_runtime_seconds - 240:
                    times_up()
                epoch_start_time = save_training_states(training_results, epoch_start_time, i, model, name, optimizer,
                                                        scheduler, start_time)
        model.last_example = -1

        valid_acc = evaluate(model, program_start_time=program_start_time)
        set_status_value(name, "completed_epochs", epoch)

        training_results.log_epoch(epoch, valid_acc, num_discarded, epoch_start_time, num_fastforward_examples)

    set_status_value(name, "finished", True)


def times_up():
    print("reached max run time. shutting down so the program can exit safely")
    status = load_status(conf.model_name)
    status["running"] = False
    save_status(conf.model_name, status)
    exit()


def save_training_states(training_results: TrainingResults, epoch_start_time, i, model, name, optimizer, scheduler,
                         start_time):

    if time.time() - start_time + 5 * 60 > conf.max_runtime_seconds != -1:
        # end 5m early before saving, as this can take long enough to send us over the max runtime
        print("reached max run time. shutting down so the program can exit safely")
        exit()
    save_time = time.time()
    print("saving model at e", training_results.epoch, "i:", i)
    model.last_example = i
    model.last_epoch = floor(training_results.epoch)
    save_model(model, optimizer, scheduler)

    save_training_results(training_results, name)
    save_time = time.time() - save_time
    epoch_start_time += save_time
    set_status_value(name, "completed_epochs", training_results.epoch)
    """upon successful saving. make a backup so that if the next save fails, we can continue from here next time"""
    duplicate_checkpoint_folder(name)

    return epoch_start_time
