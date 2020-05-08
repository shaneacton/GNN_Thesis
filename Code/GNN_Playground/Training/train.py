import time
from typing import Iterable

from torch import nn, optim

from Code.GNN_Playground.Data.Answers.extracted_answer import ExtractedAnswer
from Code.GNN_Playground.Data.Answers.one_word_answer import OneWordAnswer
from Code.GNN_Playground.Models.Vanilla.bidaf import BiDAF
from Code.GNN_Playground.Training import device, batch_size
from Datasets.Batching.batch import Batch
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

ce_loss = nn.CrossEntropyLoss()


def get_span_loss(output, batch: Batch):
    p1, p2 = output
    print(" model out - p1:",p1,"p2:",p2)
    loss_fn = lambda batch_item: ce_loss(p1, batch_item.get_start_char_index()) \
                                 + ce_loss(p2, batch_item.get_end_char_index())

    losses = [loss_fn(batch) for answer in question.answers.correct_answers]
    min_loss_value = min(losses)
    loss = [loss for loss in losses if loss.item() == min_loss_value][0]
    # given multiple correct answers, minimise the loss on the closest answer
    return loss


def get_candidate_loss(output, batch: Batch):
    answers = batch.get_answer_cand_index_vec()
    return ce_loss(output, answers)


def train_model(batches:Iterable[Batch], model: nn.Module, learning_rate=1e-3):
    model.train()
    num_epochs = 10
    max_batches = 10

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        s_time = time.time()
        total_loss = 0
        for i, batch in enumerate(batches):
            # print(training_example)
            if i >= max_batches:
                break
            output = model(batch)

            loss = 0
            optimizer.zero_grad()

            if batch.get_answer_type() == ExtractedAnswer:
                loss = get_span_loss(output, batch)
            if batch.get_answer_type() == OneWordAnswer:
                loss = get_candidate_loss(output, batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print("batch", i, "loss", loss/batch_size)
        e_time = time.time() - s_time
        sample_time = e_time/(i*batch_size)
        print("e",epoch,"loss",total_loss/batch_size,"time:",e_time,"time per sample:",sample_time)


if __name__ == "__main__":
    model = BiDAF(100).to(device)

    squad_reader = SQuADDatasetReader()
    qangaroo_reader = QUangarooDatasetReader()

    qangaroo_batch_reader = BatchReader(qangaroo_reader)
    squad_batch_reader = BatchReader(squad_reader)

    wikihop_path = QUangarooDatasetReader.dev_set_location("wikihop")
    squad_path = SQuADDatasetReader.dev_set_location()

    train_model(qangaroo_batch_reader.get_batches(wikihop_path), model)
    # train_model(squad_batch_reader.get_batches(squad_path), model)