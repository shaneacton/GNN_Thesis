import os
import sys

import torch
from transformers import LongformerForQuestionAnswering, LongformerConfig, LongformerTokenizerFast

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Datasets'))

from Code.Training import device
from Code.Training.eval_utils import evaluate
from Code.Training.metric import Metric
from Datasets.Batching.batch_reader import BatchReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

FEATURES = 600
MAX_TRAIN_BATCHES = -1
TEST_EVERY = 100
MAX_TEST_BATCHES = 30
PRINT_EVERY = 10
ATTENTION_WINDOW = 128
LAYERS = 6

if MAX_TRAIN_BATCHES > 0:
    PRINT_EVERY = min(PRINT_EVERY, MAX_TRAIN_BATCHES)


def get_ids(batch):
    context = batch.data_sample.context.get_full_context()
    question = batch.batch_items[0].question
    query = question.raw_text
    answer = question.answers
    # print("query:", query)
    # print("context:", context)
    # print("answers:", answer)
    encoding = tokenizer(query, context, return_tensors="pt")
    context_encoding = tokenizer(context)
    input_ids = encoding["input_ids"]

    start_positions_context = context_encoding.char_to_token(answer.correct_answers[0].start_char_id)
    end_positions_context = context_encoding.char_to_token(answer.correct_answers[0].end_char_id - 1)

    ctc_c = len(context_encoding["input_ids"])
    full_c = len(input_ids[0])
    q_c = full_c - ctc_c
    # print("full:", full_c, "con:", ctc_c, "q:", q_c)

    start_positions = torch.tensor([start_positions_context + q_c])
    end_positions = torch.tensor([end_positions_context + q_c])

    # print("char span:", (answer.correct_answers[0].start_char_id, answer.correct_answers[0].end_char_id - 1))
    # print("answer span:", (start_positions, end_positions))

    # the forward method will automatically set global attention on question tokens
    attention_mask = encoding["attention_mask"]
    return input_ids, attention_mask, start_positions, end_positions


def test_model(model, batch_reader):
    model.eval()
    predictions = []
    grounds = []

    with torch.no_grad():
        for b, batch in enumerate(batch_reader.get_test_batches()):
            if b > MAX_TEST_BATCHES > 0:
                break

            input_ids, attention_mask, start_positions, end_positions = get_ids(batch)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)
            start_scores, end_scores = outputs["start_logits"], outputs["end_logits"]
            for i in range(start_scores.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                answer = ' '.join(all_tokens[torch.argmax(start_scores[i]): torch.argmax(end_scores[i]) + 1])
                ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
                answer = tokenizer.decode(ans_ids)
                predictions.append(answer)
                ground_truths = [ans.raw_text for ans in batch.batch_items[i].question.answers.correct_answers]
                grounds.append(ground_truths)
                # print("predicted ans:", answer)
                # print("answer:", ground)

    print(evaluate(grounds, predictions))


def train_model(model, batch_reader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_metric = Metric("loss", print_step=True)
    model.train()

    for epoch in range(10000):

        for b, batch in enumerate(batch_reader.get_train_batches()):
            if b > MAX_TRAIN_BATCHES > 0:
                break

            input_ids, attention_mask, start_positions, end_positions = get_ids(batch)
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                         start_positions=start_positions, end_positions=end_positions)

            # print(outputs)

            loss = outputs["loss"]

            loss.backward()
            optimizer.step()

            loss_metric.report(loss.item())

            if b % PRINT_EVERY == 0:
                print(loss_metric)

            if b % TEST_EVERY == 0 and b > 0:
                test_model(model, batch_reader)

        print("Epoch", epoch, "av loss:", loss_metric.mean)
        loss_metric.flash_mean()
        test_model(model, batch_reader)


if __name__ == "__main__":
    configuration = LongformerConfig()

    configuration.attention_window = ATTENTION_WINDOW
    configuration.hidden_size = FEATURES
    configuration.intermediate_size = FEATURES
    configuration.num_labels = 2
    configuration.max_position_embeddings = 4000
    configuration.type_vocab_size = 3
    configuration.num_attention_heads = 8
    configuration.num_hidden_layers = LAYERS
    configuration.return_dict = True

    tokenizer = LongformerTokenizerFast.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
    configuration.vocab_size = tokenizer.vocab_size

    qa = LongformerForQuestionAnswering(configuration).to(device)

    print(qa)

    squad_reader = SQuADDatasetReader("SQuAD")
    squad_path = SQuADDatasetReader.train_set_location()
    squad_batch_reader = BatchReader(squad_reader, 1, squad_path)


    train_model(qa, squad_batch_reader)
