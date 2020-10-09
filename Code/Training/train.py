import os
import sys

# For importing project files

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Datasets'))


import time

import torch
from torch import optim, nn

from Code.Training import device
from Code.Config import configs, eval_conf, sysconf
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN
from Code.Training.test import test_model
from Datasets.Batching.samplebatch import SampleBatch
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader
from Code.Training.metric import Metric

ce_loss = nn.CrossEntropyLoss()

PRINT_EVERY_SAMPLES = min(eval_conf.print_stats_every_n_samples, eval_conf.max_train_batches)
if PRINT_EVERY_SAMPLES == -1:
    PRINT_EVERY_SAMPLES = eval_conf.print_stats_every_n_samples


def train_model(batch_reader: BatchReader, gnn: ContextGNN):
    gnn.train()
    num_epochs = 10

    optimizer = None

    skipped_batches_from = 0

    forward_times = Metric("forward times")
    backwards_times = Metric("backwards times")
    epoch_times = Metric("epoch times")

    loss_metric = Metric("loss", print_step=True)

    last_sample_printed_on = -PRINT_EVERY_SAMPLES

    for epoch in range(num_epochs):
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
                gnn.init_model(batch.data_sample)
                # print("initialising optim with ps:", gnn.parameters())
                optimizer = optim.Adam(gnn.parameters(), lr=eval_conf.learning_rate_base)
                print(gnn)

                optimizer.zero_grad()

            forward_start_time = time.time()
            try:
                output = gnn(batch)
            except Exception as e:
                print("Error in forward:", e)
                continue
            y = output.x

            forward_times.report(time.time() - forward_start_time)

            if batch.get_answer_type() == ExtractedAnswer:
                loss = get_span_loss(y, batch, gnn.last_batch_failures)
            if batch.get_answer_type() == CandidateAnswer:
                loss = get_candidate_loss(y, batch, gnn.last_batch_failures)

            backwards_start_time = time.time()
            loss.backward()
            optimizer.step()
            backwards_times.report(time.time() - backwards_start_time)
            loss_metric.report(float(loss.item()))

            samples = b * batch.batch_size
            i += 1
            if samples - last_sample_printed_on >= PRINT_EVERY_SAMPLES and PRINT_EVERY_SAMPLES != -1:
                # print("y:", y, "shape:", y.size())
                print("\nbatch", b, loss_metric)
                if sysconf.print_times:
                    embedding_times = gnn.embedder.embedding_times
                    # estimate of the total time spent not in encoding
                    model_times = forward_times - embedding_times * y.size(0)
                    model_times.name = "model time"
                    total_times = model_times + backwards_times + embedding_times
                    total_times.name = "total time"
                    print("\t", model_times, "\n\t", backwards_times, "\n\t", embedding_times, "\n\t", total_times)
                last_sample_printed_on = samples

            skipped_batches_from = 0  # has not skipped

        epoch_times.report(time.time() - epoch_start_time)
        print("-----------\te", epoch, "\t-----------------")

        test_model(qangaroo_batch_reader, gnn)


def get_span_loss(output, batch: SampleBatch):
    p1, p2 = output
    answers = batch.get_answers_tensor()
    return ce_loss(p1, answers[:,0]) + ce_loss(p2, answers[:,1])


def get_candidate_loss(output, batch: SampleBatch, failures):
    answers = batch.get_answers_tensor()
    if len(failures) > 0:
        # had failures, must remove answers from failed samples
        batch_size = len(batch.batch_items)
        successes = set(list(range(batch_size))).difference(set(failures))
        successes = torch.tensor(list(successes)).to(device)
        answers = torch.index_select(answers, dim=0, index=successes)
        # print("found failures:",failures, "successes:",successes)

    # print("answers:", answers.size(), "output:", output.size())
    # output = output.view(1, -1)
    # print("answers:", answers, "\nout:", output)
    return ce_loss(output, answers)


if __name__ == "__main__":
    gnn = configs.get_gnn()

    squad_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    wikihop_path = QUangarooDatasetReader.dev_set_location("wikihop")
    squad_path = SQuADDatasetReader.dev_set_location()

    qangaroo_batch_reader = BatchReader(qangaroo_reader, eval_conf.batch_size, wikihop_path)
    squad_batch_reader = BatchReader(squad_reader, eval_conf.batch_size, squad_path)

    train_model(qangaroo_batch_reader, gnn)
    # train_model(squad_batch_reader, model)