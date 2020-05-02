from typing import Iterable

import torch
from torch import nn, optim

from Code.GNN_Playground.Data.Answers.extracted_answer import ExtractedAnswer
from Code.GNN_Playground.Data.Answers.one_word_answer import OneWordAnswer
from Code.GNN_Playground.Data.question import Question
from Code.GNN_Playground.Data.training_example import TrainingExample
from Code.GNN_Playground.Models.Vanilla.bidaf import BiDAF
from Code.GNN_Playground.Training import device
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

ce_loss = nn.CrossEntropyLoss()


def get_span_loss(output, question):
    p1, p2 = output
    losses = [ce_loss(p1, answer.start_token_id) + ce_loss(p2, answer.end_token_id)
              for answer in question.answers.correct_answers]
    min_loss_value = min(losses)
    loss = [loss for loss in losses if loss.item() == min_loss_value][0]
    # given multiple correct answers, minimise the loss on the closest answer
    return loss


def get_candidate_loss(output, question: Question):
    answers = question.get_answer_cand_vec()
    return ce_loss(output, answers)


def train_model(training_data:Iterable[TrainingExample], model: nn.Module, learning_rate=1e-3):
    model.train()
    num_epochs = 10
    max_batches = 10

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for i, training_example in enumerate(training_data):
            # print(training_example)
            if i >= max_batches:
                break
            for question in training_example.questions:
                output = model(training_example.context, question)

                loss = 0
                optimizer.zero_grad()

                if question.get_answer_type() == ExtractedAnswer:
                    loss = get_span_loss(output, question)
                if question.get_answer_type() == OneWordAnswer:
                    loss = get_candidate_loss(output, question)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("batch",i,"loss",loss)
        print("e",epoch,"loss",total_loss)


if __name__ == "__main__":
    model = BiDAF(100).to(device)

    squad_reader = SQuADDatasetReader()
    qangaroo_reader = QUangarooDatasetReader()

    # training_iterator = squad_reader.get_training_examples(SQuADDatasetReader.dev_set_location())
    training_iterator = qangaroo_reader.get_training_examples(QUangarooDatasetReader.dev_set_location("wikihop"))
    train_model(training_iterator,model)