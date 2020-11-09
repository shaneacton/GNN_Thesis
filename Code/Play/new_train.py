import os
import sys

import nlp
import torch
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Play.encoding import TextEncoder
from Code.Training.eval_utils import evaluate
from Code.Play.initialiser import get_trainer, get_composite_span_longformer

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
encoder = TextEncoder(tokenizer)

TRAIN = 'train_data.pt'
VALID = 'valid_data.pt'
OUT = "models"

DATASET = "squad"  # "qangaroo"  # "squad"
VERSION = None  # "wikihop"
# DATASET = "qangaroo"  # "qangaroo"  # "squad"
# VERSION = "wikihop"


def data_loc(set_name):
    data_name = VERSION if VERSION else DATASET
    return os.path.join(data_name, set_name)


def evaluate_model(model, valid_dataset):
    model = model.cuda()
    model.eval()

    dataloader = DataLoader(valid_dataset, batch_size=2)

    answers = []
    with torch.no_grad():
        for batch in nlp.tqdm(dataloader):
            print("batch:", batch)
            start_scores, end_scores = model(batch)
            print("starts:", start_scores.size(), "ends:", end_scores.size())
            for i in range(start_scores.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                answer = ' '.join(all_tokens[torch.argmax(start_scores[i]): torch.argmax(end_scores[i]) + 1])
                ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
                answer = tokenizer.decode(ans_ids)
                answers.append(answer)
                print("got answers:", answer)
    print("got answers")

    predictions = []
    references = []
    valid_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.VALIDATION, name=VERSION)  # raw text version

    for ref, pred in zip(valid_dataset, answers):
        predictions.append(pred)
        # print("ref:", ref)
        references.append(ref['answers']['text'])

    print(evaluate(references, predictions))


print("starting model init")
# model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
from Code.Config import gec, gnnc
from Code.Config import gcc

embedder = gec.get_graph_embedder(gcc)

gat = ContextGAT(embedder, gnnc)
# Get datasets
print('loading data')
train_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.TRAIN, name=VERSION)
valid_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.VALIDATION, name=VERSION)
print('loading done')

evaluate_model(gat, valid_dataset)

# trainer = get_trainer(model, data_loc(OUT), train_dataset, valid_dataset)
#
#
# def get_latest_model():
#     out = os.path.join(".", data_loc(OUT))
#     checks = [c for c in os.listdir(out) if "check" in c]
#     if len(checks) == 0:
#         return None
#     steps = [int(c.split("-")[1]) for c in checks]
#     hi=-1
#     max_i = -1
#     for i in range(len(steps)):
#         if steps[i] > hi:
#             hi=steps[i]
#             max_i = i
#     return checks[max_i]
#
#
# check = get_latest_model()
# check = None if check is None else os.path.join(".", data_loc(OUT), check)
# print("checkpoint:", check)
# trainer.train(model_path=check)
# trainer.save_model()
#
# evaluate_model(model, valid_dataset)
