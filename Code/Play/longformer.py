import itertools
import os
import sys
import time
from typing import Tuple

import torch
from pandas import np
from transformers import LongformerConfig, LongformerModel

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Datasets'))

from Code.Config import eval_conf, configs, sysconf, gec
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Data.Text.pretrained_token_sequence_embedder import PretrainedTokenSequenceEmbedder
from Code.Models.GNNs.OutputModules.node_selection import NodeSelection
from Code.Training import device
from Code.Training.metric import Metric
from Code.Training.train import ce_loss, PRINT_EVERY_SAMPLES
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Batching.samplebatch import SampleBatch
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader


def train_model(batch_reader: BatchReader, model):
    model.train()

    skipped_batches_from = 0

    forward_times = Metric("forward times")
    backwards_times = Metric("backwards times")
    epoch_times = Metric("epoch times")

    loss_metric = Metric("loss", print_step=True)

    last_sample_printed_on = -PRINT_EVERY_SAMPLES

    selector = NodeSelection(768).to(device)

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), selector.parameters()), lr=eval_conf.learning_rate_base)

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

            optimizer.zero_grad()

            forward_start_time = time.time()
            # try:
            ids, type_ids, global_attention_mask, starts = get_ids(batch)

            try:
                y, _ = model(input_ids=ids, token_type_ids=type_ids, global_attention_mask=global_attention_mask)
                probs = selector.get_probabilities(y.squeeze(), starts)
            except:
                continue
            # print("out:", y.size(), "probs:", probs.size())


            forward_times.report(time.time() - forward_start_time)

            if batch.get_answer_type() == ExtractedAnswer:
                loss = get_span_loss(probs, batch)
            if batch.get_answer_type() == CandidateAnswer:
                loss = get_candidate_loss(probs, batch)

            # except Exception as e:
            #     continue


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
                    # some of the batch items may have failed
                    batch_size = y[0].size(0) if isinstance(y, Tuple) else y.size(0)
                    # estimate of the total time spent not in encoding
                    model_times = forward_times
                    model_times.name = "model time"
                    total_times = model_times + backwards_times
                    total_times.name = "total time"
                    total_times.print_total = True
                    print("\t", model_times, "\n\t", backwards_times, "\n\t", total_times)
                last_sample_printed_on = samples

            skipped_batches_from = 0  # has not skipped

        epoch_times.report(time.time() - epoch_start_time)
        test_model(batch_reader, model, selector)
        print("-----------\te", epoch, "\t-----------------")


def get_ids(batch):
    q = batch.batch_items[0].question
    q_ids = indexer(q.raw_text)
    cands = [cand.raw_text for cand in q.answers.answer_candidates]
    cands = [indexer(cand) for cand in cands]
    starts = []
    s = 0
    for cand in cands:
        starts.append(s)
        s += cand.size(1)
    starts = torch.tensor(starts).view(1, -1)
    cand_ids = torch.cat(cands, dim=1)

    ctx_ids = indexer(batch.data_sample.context.get_full_context())
    sep_id = torch.tensor(configuration.sep_token_id).view(1, 1)
    # print("ctx:", ctx_ids.size(), "q:", q_ids.size(), "sep:", sep_id)
    # print("cands:", cand_ids.size(), "cand starts:", starts)
    ids = torch.cat([cand_ids, sep_id, q_ids, sep_id, ctx_ids], dim=1)
    # print("ids:", ids.size())
    type_ids = torch.cat([torch.tensor([0] * cand_ids.size(1)), torch.tensor([1] * (q_ids.size(1) + 2)),
                          torch.tensor([2] * ctx_ids.size(1))])

    global_attention_mask = torch.zeros(ids.shape, dtype=torch.long, device=device)
    global_ids = list(range(cand_ids.size(1) + 1 + q_ids.size(1)))  # all cands and q's
    global_attention_mask[:, global_ids] = 1
    return ids, type_ids, global_attention_mask, starts


def test_model(batch_reader: BatchReader, model, selector):
    gnn.eval()

    total_acc = 0
    total_chance = 0
    count = 0
    with torch.no_grad():

        for b, batch in enumerate(batch_reader.get_test_batches()):
            if 0 < eval_conf.max_test_batches < count:
                break

            try:
                ids, type_ids, global_attention_mask, starts = get_ids(batch)

                y, _ = model(input_ids=ids, token_type_ids=type_ids, global_attention_mask=global_attention_mask)
                y = selector.get_probabilities(y.squeeze(), starts)

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
    gnn = configs.get_gnn()

    squad_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    wikihop_path = QUangarooDatasetReader.train_set_location("wikihop")
    squad_path = SQuADDatasetReader.train_set_location()

    qangaroo_batch_reader = BatchReader(qangaroo_reader, eval_conf.batch_size, wikihop_path)
    squad_batch_reader = BatchReader(squad_reader, eval_conf.batch_size, squad_path)

    # Initializing a Longformer configuration
    configuration = LongformerConfig()
    token_seq_embedder = PretrainedTokenSequenceEmbedder(gec)
    indexer = lambda string: token_seq_embedder.index(token_seq_embedder.tokenise(string))

    vocab_size = len(token_seq_embedder.bert_tokeniser.vocab)
    configuration.vocab_size = vocab_size
    configuration.max_position_embeddings = 4000
    configuration.num_hidden_layers = 5
    configuration.attention_window = 50
    configuration.type_vocab_size = 3
    print("attention window:", configuration.attention_window)

    # Initializing a model from the configuration
    model = LongformerModel(configuration)
    print("num hidden layers:", configuration.num_hidden_layers)

    train_model(qangaroo_batch_reader, model)


