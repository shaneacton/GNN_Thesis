from typing import Tuple

import torch

from Code.Config import eval_conf
from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN
from Datasets.Batching.batch_reader import BatchReader
import numpy as np


def test_model(batch_reader: BatchReader, gnn: ContextGNN):
    gnn.eval()

    total_acc = 0
    total_chance = 0
    count = 0
    with torch.no_grad():

        for b, batch in enumerate(batch_reader.get_test_batches()):
            if 0 < eval_conf.max_test_batches < count:
                break

            for batch_item in batch.batch_items:

                sample = batch_item.data_sample

                try:
                    output = gnn(sample)

                    y = output.x
                    # print("y(", y.size(), "):", y)
                    answers = batch.get_answers_tensor()
                except Exception as e:
                    continue

                if isinstance(y, Tuple):
                    p1, p2 = np.argmax(y[0].cpu(), axis=1), np.argmax(y[1].cpu(), axis=1)
                    if p1 == answers[:, 0]:
                        total_acc += 1
                    if p2 == answers[:, 1]:
                        total_acc += 1
                    count += 2  # 2 per example

                if not isinstance(y, Tuple):
                    # chance not relevant in span selection
                    total_chance += 1.0 / y.size(1)
                    predictions = np.argmax(y.cpu(), axis=1)

                    if answers == predictions:
                        # print("+++++ correct:", answers, predictions, "++++++++++++++")
                        total_acc += 1
                    count += 1

    accuracy = total_acc / count
    chance = total_chance / count
    print("accuracy:", accuracy, "count:", count, "chance:", chance)

    return accuracy
