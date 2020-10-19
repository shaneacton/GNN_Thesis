import itertools
import os
import sys
import time
from typing import Tuple

import numpy as np
import torch
from transformers import LongformerConfig, LongformerModel

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Datasets'))

from Code.Models.Transformers.context_transformer import ContextTransformer
from Code.Config import eval_conf, sysconf
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Training.metric import Metric
from Code.Training.train import ce_loss, PRINT_EVERY_SAMPLES
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Batching.samplebatch import SampleBatch
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

FEATURES = 408

distance = Metric("distance", beta=10)


def train_model(batch_reader: BatchReader, model):
    model.train()

    skipped_batches_from = 0

    forward_times = Metric("forward times")
    backwards_times = Metric("backwards times")
    epoch_times = Metric("epoch times")

    loss_metric = Metric("loss", print_step=True)

    last_sample_printed_on = -PRINT_EVERY_SAMPLES

    optimizer = None

    for epoch in range(eval_conf.num_epochs):
        epoch_start_time = time.time()

        i = 0  # number of batches used so far since skip
        # print("batches:", list(batch_reader.get_all_batches()))
        for b, batch in enumerate(batch_reader.get_train_batches()):
            # b ~ the batch id
            if b < skipped_batches_from:
                # last epoch was cut short by max_batches, continue where it left off
                continue

            if i >= eval_conf.max_train_batches != -1:
                # this epoch has hit its max_batches
                # print("skipping batches from", b, "in epoch", epoch)
                skipped_batches_from = b
                break

            if optimizer is None:
                # gnn must see a data sample to initialise. optim must wait
                model.init_output_model(batch.data_sample, FEATURES)
                # print("initialising optim with ps:", gnn.parameters())
                optimizer = torch.optim.Adam(model.parameters(), lr=eval_conf.learning_rate_base)
                print(model)

            optimizer.zero_grad()

            forward_start_time = time.time()

            try:
                y = model(batch)

                forward_times.report(time.time() - forward_start_time)

                if batch.get_answer_type() == ExtractedAnswer:
                    loss = get_span_loss(y, batch)
                if batch.get_answer_type() == CandidateAnswer:
                    loss = get_candidate_loss(y, batch)

            except Exception as e:
                # print(e)
                continue

            backwards_start_time = time.time()
            loss.backward()
            optimizer.step()
            backwards_times.report(time.time() - backwards_start_time)
            loss_metric.report(float(loss.item()))
            samples = b * batch.batch_size
            i += 1
            if samples - last_sample_printed_on >= PRINT_EVERY_SAMPLES and PRINT_EVERY_SAMPLES != -1:
                print("\nbatch", b, loss_metric)
                if sysconf.print_times:
                    model_times = forward_times
                    model_times.name = "model time"
                    total_times = model_times + backwards_times
                    total_times.name = "total time"
                    total_times.print_total = True
                    print("\t", model_times, "\n\t", backwards_times, "\n\t", total_times)
                last_sample_printed_on = samples

            skipped_batches_from = 0  # has not skipped

        epoch_times.report(time.time() - epoch_start_time)
        test_model(batch_reader, model)
        print("-----------\te", epoch, "\t-----------------")


def test_model(batch_reader: BatchReader, model):
    model.eval()

    total_acc = 0
    total_chance = 0
    count = 0
    norm_dist = 0
    with torch.no_grad():

        for b, batch in enumerate(batch_reader.get_test_batches()):
            if 0 < eval_conf.max_test_batches < count:
                break

            # try:
            y = model(batch)

            answers = batch.get_answers_tensor()

            if isinstance(y, Tuple):
                # print("using span acc")
                p1, p2 = np.argmax(y[0].cpu(), axis=1), np.argmax(y[1].cpu(), axis=1)
                # print("p1:", p1.item(), "p2:", p2.item(), "y:", y)
                dist = abs(answers[:, 0].item() - p1.item()) + abs(answers[:, 1].item() - p2.item())
                dist /= y[0].size(1)
                if p1 == answers[:, 0]:
                    total_acc += 1
                if p2 == answers[:, 1]:
                    total_acc += 1
                norm_dist += dist

            if not isinstance(y, Tuple):
                # chance not relevant in span selection
                # print("y(", y.size(), "):", y)

                total_chance += 1.0 / y.size(1)
                predictions = np.argmax(y.cpu(), axis=1)

                if answers == predictions:
                    # print("+++++ correct:", answers, predictions, "++++++++++++++")
                    total_acc += 1

            # except Exception as e:
            #     print(e)
            #     continue

            count += 1

    accuracy = total_acc / count
    chance = total_chance / count
    norm_dist /= count
    distance.report(norm_dist)

    if norm_dist != 0:
        print("accuracy:", accuracy, "count:", count, "norm dist:", norm_dist, distance)
    else:
        print("accuracy:", accuracy, "count:", count, "chance:", chance)

    model.train()
    return accuracy


def get_span_loss(output, batch: SampleBatch):
    #todo implement failures for batching
    p1, p2 = output
    answers = batch.get_answers_tensor()
    # print("p1:", p1, "p2:", p2, "ans:", answers)
    return ce_loss(p1, answers[:,0]) + ce_loss(p2, answers[:,1])


def get_candidate_loss(output, batch: SampleBatch):
    answers = batch.get_answers_tensor()

    # print("answers:", answers.size(), "output:", output.size())
    # output = output.view(1, -1)
    # print("answers:", answers, "\nout:", output)
    return ce_loss(output, answers)


if __name__ == "__main__":
    squad_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    wikihop_path = QUangarooDatasetReader.train_set_location("wikihop")
    squad_path = SQuADDatasetReader.train_set_location()

    qangaroo_batch_reader = BatchReader(qangaroo_reader, eval_conf.batch_size, wikihop_path)
    squad_batch_reader = BatchReader(squad_reader, eval_conf.batch_size, squad_path)

    # Initializing a Longformer configuration
    configuration = LongformerConfig()

    configuration.attention_window = 50
    configuration.hidden_size = FEATURES
    configuration.intermediate_size=FEATURES

    model = ContextTransformer(LongformerModel, configuration, 5)

    train_model(squad_batch_reader, model)


