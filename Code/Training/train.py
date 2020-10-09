import os
import sys
import time

from torch import optim, nn

# For importing project files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Datasets'))

from Code.Config import configs, eval_conf, sysconf
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN
from Code.Training.test import test_model
from Datasets.Batching.samplebatch import SampleBatch
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

ce_loss = nn.CrossEntropyLoss()

PRINT_BATCH_EVERY = min(eval_conf.print_batch_every, eval_conf.max_train_batches)
if PRINT_BATCH_EVERY == -1:
    PRINT_BATCH_EVERY = eval_conf.print_batch_every


def train_model(batch_reader: BatchReader, gnn: ContextGNN):
    gnn.train()
    num_epochs = 10

    optimizer = None

    skipped_batches_from = 0

    rolling_average = -1

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0

        i = 0  # number of batches used so far
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

            forward_time = time.time() - forward_start_time

            if batch.get_answer_type() == ExtractedAnswer:
                loss = get_span_loss(y, batch)
            if batch.get_answer_type() == CandidateAnswer:
                loss = get_candidate_loss(y, batch)


            backwards_start_time = time.time()
            loss.backward()
            optimizer.step()
            backwards_time = time.time() - backwards_start_time
            loss_val = float(loss.item())
            total_loss += loss_val
            if rolling_average == -1:
                rolling_average = loss_val
            else:
                if i < 5:
                    a = 0.9 # converge quickly on a rough average
                elif i < 50 and epoch == 0:
                    a = 0.95  # converge quickly on a rough average
                else:
                    a = 0.998
                rolling_average = a * rolling_average + (1-a) * loss_val
            if i % PRINT_BATCH_EVERY == 0 and PRINT_BATCH_EVERY != -1:
                # print("y:", y, "shape:", y.size())
                if sysconf.print_times:
                    print("forward time:", forward_time, "backwards time:", backwards_time)
                print("batch", i, "loss", loss_val / batch.batch_size, "rolling loss:\t",rolling_average, "\n")

            i += 1
            skipped_batches_from = 0  # has not skipped

        e_time = time.time() - epoch_start_time
        num_samples = i * batch.batch_size
        sample_time = e_time / num_samples
        sample_loss = total_loss / num_samples
        print("e", epoch, "loss per sample:", sample_loss, "time:", e_time,
              "time per sample:", sample_time, "num samples:", num_samples)

        accuracy = test_model(qangaroo_batch_reader, gnn)


def get_span_loss(output, batch: SampleBatch):
    p1, p2 = output
    answers = batch.get_answers_tensor()
    return ce_loss(p1, answers[:,0]) + ce_loss(p2, answers[:,1])


def get_candidate_loss(output, batch: SampleBatch):
    answers = batch.get_answers_tensor()
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