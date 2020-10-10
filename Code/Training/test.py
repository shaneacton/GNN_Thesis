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
                except Exception as e:
                    # print("Error in forward:", e)
                    continue

                y = output.x
                # print("y(", y.size(), "):", y)
                total_chance += 1.0 / y.size(1)

                predictions = np.argmax(y.cpu(), axis=1)

                answers = batch.get_answers_tensor()

                # print("predictions:", predictions)
                # print("answers:", answers)

                # acc = accuracy_score(answers.cpu(), predictions)
                acc = 0
                if answers == predictions:
                    # print("+++++ correct:", answers, predictions, "++++++++++++++")
                    acc = 1
                total_acc += acc
                count += 1

    accuracy = total_acc / count
    chance = total_chance / count
    print("accuracy:", accuracy, "count:", count, "chance:", chance)

    return accuracy
