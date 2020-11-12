import os
import sys

import nlp
import torch
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering

from Code.Data.Text.text_utils import context, question
from Code.Play.text_and_tensor_coalator import composite_data_collator

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Play.text_encoder import TextEncoder
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


def save_dataset():

    # load train and validation split of squad
    train_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.TRAIN, name=VERSION)
    valid_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.VALIDATION, name=VERSION)

    # dataloader = DataLoader(valid_dataset, batch_size=16)
    # for batch in nlp.tqdm(dataloader):
    #     print("before:", batch)
    #     break

    print("mapping dataset")
    train_dataset = train_dataset.map(encoder.get_basic_features)
    valid_dataset = valid_dataset.map(encoder.get_basic_features) #, load_from_cache_file=False)

    # dataloader = DataLoader(valid_dataset, batch_size=16)
    # for batch in nlp.tqdm(dataloader):
    #     print("after", batch)
    #     break

    # set the tensor type and the columns which the dataset should return
    columns = ['start_positions', 'end_positions']
    train_dataset.set_format(type='torch', columns=columns, output_all_columns=True)
    valid_dataset.set_format(type='torch', columns=columns, output_all_columns=True)
    # columns = ['context', 'question']
    # train_dataset.set_format(type=None, columns=columns)
    # valid_dataset.set_format(type=None, columns=columns)

    torch.save(train_dataset, data_loc(TRAIN))
    torch.save(valid_dataset, data_loc(VALID))


def evaluate_model(model, valid_dataset):
    model = model.cuda()
    model.eval()

    dataloader = DataLoader(valid_dataset, batch_size=16)
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

    predictions = []
    with torch.no_grad():
        for batch in nlp.tqdm(dataloader):
            # print("batch:", batch)
            _, start_scores, end_scores = model(batch)
            # print("start probs:", start_scores, "\n:end probs:", end_scores)
            # break

            for i in range(start_scores.shape[0]):
                """for each batch item, """
                qa = tokenizer(question(batch)[i], context(batch)[i], pad_to_max_length=True, max_length=512, truncation=True)
                s, e = torch.argmax(start_scores[i]), torch.argmax(end_scores[i]) + 1
                predicted = ' '.join(qa.tokens()[s: e])
                ans_ids = tokenizer.convert_tokens_to_ids(predicted.split())
                predicted = tokenizer.decode(ans_ids)
                # print("(s,e):", (s, e), "pred:", predicted, "total tokens:", len(qa.tokens()), "\n\n")
                predictions.append(predicted)
    #         raise Exception()
    # print("got answers")

    predictions = []
    references = []
    valid_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.VALIDATION, name=VERSION)  # raw text version

    for ref, pred in zip(valid_dataset, predictions):
        predictions.append(pred)
        # print("ref:", ref)
        references.append(ref['answers']['text'])

    print(evaluate(references, predictions))


print("starting model init")
from Code.Config import gec, gnnc
from Code.Config import gcc

embedder = gec.get_graph_embedder(gcc)

gat = ContextGAT(embedder, gnnc)
# Get datasets
print('loading data')
# save_dataset()
train_dataset = torch.load(data_loc(TRAIN))
valid_dataset = torch.load(data_loc(VALID))
print('loading done')

evaluate_model(gat, valid_dataset)
# evaluate_model(None, valid_dataset)


trainer = get_trainer(gat, data_loc(OUT), train_dataset, valid_dataset)
trainer.data_collator = composite_data_collator
#
#
def get_latest_model():
    out = os.path.join(".", data_loc(OUT))
    checks = [c for c in os.listdir(out) if "check" in c]
    if len(checks) == 0:
        return None
    steps = [int(c.split("-")[1]) for c in checks]
    hi=-1
    max_i = -1
    for i in range(len(steps)):
        if steps[i] > hi:
            hi=steps[i]
            max_i = i
    return checks[max_i]
#
#
check = get_latest_model()
check = None if check is None else os.path.join(".", data_loc(OUT), check)
print("checkpoint:", check)
trainer.train(model_path=check)
trainer.save_model()

evaluate_model(gat, valid_dataset)
