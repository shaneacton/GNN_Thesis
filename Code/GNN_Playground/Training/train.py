from typing import Iterable

from torch import nn, optim

from Code.GNN_Playground.Data.Answers.extracted_answer import ExtractedAnswer
from Code.GNN_Playground.Data.training_example import TrainingExample

span_crit = nn.CrossEntropyLoss()


def train_span_model(context, question, model: nn.Module, optimizer):
    p1, p2 = model(context, question)
    optimizer.zero_grad()
    loss = min([span_crit(p1, answer.start_id) + span_crit(p2, answer.end_id)
                for answer in question.answers.correct_answers])
    # given multiple correct answers, minimise the loss on the closest answer
    loss.backward()
    optimizer.step()


def train_model(training_data:Iterable[TrainingExample], model: nn.Module, learning_rate=1e-3):
    model.train()
    num_epochs = 10

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, training_example in enumerate(training_data):
            for question in training_example.questions:
                if question.get_answer_type() == ExtractedAnswer:
                    """
                        this is a span prediction problem
                    """
                    train_span_model(training_example.context, question, model, optimizer)