import time

from torch import nn, optim

from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Models.RNN.bidaf import BiDAF
from Code.Models.qa_model import QAModel
from Code.Training import device
from Datasets.Batching.batch import Batch
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

ce_loss = nn.CrossEntropyLoss()
MAX_BATCHES = -1
PRINT_BATCH_EVERY = 50

def get_span_loss(output, batch: Batch):
    p1, p2 = output
    answers = batch.get_answers()
    return ce_loss(p1, answers[:,0]) + ce_loss(p2, answers[:,1])


def get_candidate_loss(output, batch: Batch):
    answers = batch.get_answers()
    return ce_loss(output, answers)


def train_model(batch_reader: BatchReader, model: QAModel, learning_rate=1e-3):
    model.train()
    num_epochs = 10

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("training model",model, "with", model.num_params,"params")

    for epoch in range(num_epochs):
        s_time = time.time()
        total_loss = 0
        for i, batch in enumerate(batch_reader.get_batches()):
            # print(batch)
            if i >= MAX_BATCHES and MAX_BATCHES != -1:
                break
            optimizer.zero_grad()
            output = model(batch)

            loss = 0

            if batch.get_answer_type() == ExtractedAnswer:
                loss = get_span_loss(output, batch)
            if batch.get_answer_type() == CandidateAnswer:
                loss = get_candidate_loss(output, batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i%PRINT_BATCH_EVERY == 0 and PRINT_BATCH_EVERY != -1:
                print("batch", i, "loss", loss/batch.batch_size)

        e_time = time.time() - s_time
        num_samples = i*batch.batch_size
        sample_time = e_time/num_samples
        sample_loss = total_loss/num_samples
        print("e",epoch,"loss per sample:",sample_loss,"time:",e_time,
              "time per sample:",sample_time, "num samples:", num_samples)


if __name__ == "__main__":
    model = BiDAF(100).to(device)

    squad_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    wikihop_path = QUangarooDatasetReader.dev_set_location("wikihop")
    squad_path = SQuADDatasetReader.dev_set_location()

    qangaroo_batch_reader = BatchReader(qangaroo_reader, 1, wikihop_path)
    squad_batch_reader = BatchReader(squad_reader, 10, squad_path)

    train_model(qangaroo_batch_reader, model)
    # train_model(squad_batch_reader, model)
