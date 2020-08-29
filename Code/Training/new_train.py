import time

from torch import optim, nn

from Code.Config import configs
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Models.GNNs.context_gnn import ContextGNN
from Datasets.Batching.batch import Batch
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

ce_loss = nn.CrossEntropyLoss()

MAX_BATCHES = -1
PRINT_BATCH_EVERY = 10


def train_model(batch_reader: BatchReader, gnn: ContextGNN, learning_rate=1e-3):
    gnn.train()
    num_epochs = 10

    optimizer = None

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        rolling_average = -1
        for i, batch in enumerate(batch_reader.get_batches()):
            # print(batch)
            if i >= MAX_BATCHES and MAX_BATCHES != -1:
                break
            for batch_item in batch.batch_items:
                sample = batch_item.data_sample

                if optimizer is None:
                    # gnn must see a data sample to initialise. optim must wait
                    gnn.init_model(sample)
                    # print("initialising optim with ps:", gnn.parameters())
                    optimizer = optim.Adam(gnn.parameters(), lr=learning_rate)

                optimizer.zero_grad()
                forward_start_time = time.time()
                try:
                    output = gnn(sample)
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
                    a = 0.98
                    rolling_average = a * rolling_average + (1-a) * loss_val
                if i % PRINT_BATCH_EVERY == 0 and PRINT_BATCH_EVERY != -1:
                    print("y:", y, "shape:", y.size())
                    print("forward time:", forward_time, "backwards time:", backwards_time)
                    print("batch", i, "loss", loss_val / batch.batch_size, "rolling loss:",rolling_average)


        e_time = time.time() - epoch_start_time
        num_samples = i * batch.batch_size
        sample_time = e_time / num_samples
        sample_loss = total_loss / num_samples
        print("e", epoch, "loss per sample:", sample_loss, "time:", e_time,
              "time per sample:", sample_time, "num samples:", num_samples)


def get_span_loss(output, batch: Batch):
    p1, p2 = output
    answers = batch.get_answers_tensor()
    return ce_loss(p1, answers[:,0]) + ce_loss(p2, answers[:,1])


def get_candidate_loss(output, batch: Batch):
    answers = batch.get_answers_tensor()
    # print("answers:", answers.size())
    output = output.view(1, -1)
    return ce_loss(output, answers)


if __name__ == "__main__":
    gnn = configs.get_gnn()

    squad_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    wikihop_path = QUangarooDatasetReader.dev_set_location("wikihop")
    squad_path = SQuADDatasetReader.dev_set_location()

    qangaroo_batch_reader = BatchReader(qangaroo_reader, 1, wikihop_path)
    squad_batch_reader = BatchReader(squad_reader, 1, squad_path)

    train_model(qangaroo_batch_reader, gnn)
    # train_model(squad_batch_reader, model)