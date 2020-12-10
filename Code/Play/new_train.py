import os
import sys
from os.path import exists

import nlp
import torch
from torch.utils.data import DataLoader


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Models.GNNs.OutputModules.candidate_selection import CandidateSelection
from Code.Models.GNNs.OutputModules.span_selection import SpanSelection
from Code.Data.Text.text_utils import candidates
from Code.Play.text_and_tensor_coalator import composite_data_collator
from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Play.text_encoder import TextEncoder
from Code.Training.eval_utils import evaluate
from Code.Play.initialiser import get_trainer, get_tokenizer

from Code.Config import gec, gnnc
from Code.Config import gcc


TRAIN = 'train_data.pt'
VALID = 'valid_data.pt'
OUT = "context_model"

DATASET = "squad"  # "qangaroo"  # "squad"
VERSION = None  # "wikihop"
# DATASET = "qangaroo"  # "qangaroo"  # "squad"
# VERSION = "wikihop"


def data_loc(set_name):
    data_name = VERSION if VERSION else DATASET
    return os.path.join(data_name, set_name)


def load_dataset(split):
    remaining_tries = 100
    dataset = None
    e = None
    while remaining_tries > 0:
        """load dataset from online"""
        try:
            dataset = nlp.load_dataset(path=DATASET, split=split, name=VERSION)
            break  # loaded successfully
        except Exception as e:
            remaining_tries -= 1  # retry
            if remaining_tries == 0:
                print("failed to load datasets though network")
                raise e

    return dataset


def process_dataset():
    if exists(data_loc(VALID)):
        """already saved"""
        return
    # load train and validation split of squad
    train_dataset = load_dataset(nlp.Split.TRAIN)
    valid_dataset = load_dataset(nlp.Split.VALIDATION)
    encoder = TextEncoder(get_tokenizer())

    print("mapping dataset")
    train_dataset = train_dataset.map(encoder.get_processed_example, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(encoder.get_processed_example, load_from_cache_file=False)

    dataloader = DataLoader(valid_dataset, batch_size=1)
    batch = None
    for batch in nlp.tqdm(dataloader):
        # print("after", batch)
        break

    # set the tensor type and the columns which the dataset should return
    if 'start_positions' in batch and 'end_positions' in batch:
        tensor_columns = ['start_positions', "end_positions"]
    elif "answer" in batch:
        tensor_columns = ['answer']
    else:
        raise Exception()

    train_dataset.set_format(type='torch', columns=tensor_columns, output_all_columns=True)
    valid_dataset.set_format(type='torch', columns=tensor_columns, output_all_columns=True)

    torch.save(train_dataset, data_loc(TRAIN))
    torch.save(valid_dataset, data_loc(VALID))


def get_data_sample():
    dataset = torch.load(data_loc(TRAIN))
    dataloader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for example in nlp.tqdm(dataloader):
            return example


def evaluate_model(model, valid_dataset):
    model = model.cuda()
    model.eval()

    batch_size = 1
    dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    tokenizer = get_tokenizer()
    valid_dataset = load_dataset(nlp.Split.VALIDATION)

    encoder = TextEncoder(get_tokenizer())

    answers = []
    with torch.no_grad():
        for batch in nlp.tqdm(dataloader):
            if isinstance(model.output_model, SpanSelection):
                _, start_scores, end_scores = model(batch)
                # print("start probs:", start_scores, "\n:end probs:", end_scores)
            elif isinstance(model.output_model, CandidateSelection):
                _, probs = model(batch)
                # print("probs:", probs, "cands:", candidates(batch), "ans:", batch['answer'], "q:", question(batch))
            else:
                raise Exception("unsupported output model: " + repr(model.output_model))

            for i in range(batch_size):
                """for each batch item, get the string prediction of the model"""
                qa = encoder.get_encoding(batch)
                if isinstance(model.output_model, SpanSelection):
                    s, e = torch.argmax(start_scores[i]), torch.argmax(end_scores[i]) + 1
                    predicted = ' '.join(qa.tokens()[s: e])
                    ans_ids = tokenizer.convert_tokens_to_ids(predicted.split())
                    predicted = tokenizer.decode(ans_ids)
                else:
                    c = torch.argmax(probs[i])
                    predicted = candidates(batch).split('</s>')[c]
                    # print("predicted:", predicted)
                # print("(s,e):", (s, e), "pred:", predicted, "total tokens:", len(qa.tokens()), "\n\n")
                answers.append(predicted)
    #         raise Exception()
    # print("got answers")

    predictions = []
    references = []
    for ref, pred in zip(valid_dataset, answers):
        predictions.append(pred)
        # print("ref:", ref)
        references.append(ref['answers']['text'])

    print(evaluate(references, predictions))


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


if __name__ == "__main__":
    print("starting model init")
    embedder = gec.get_graph_embedder(gcc)

    gat = ContextGAT(embedder, gnnc)
    # print("data sample:", get_data_sample())
    # Get datasets
    print('loading data')
    process_dataset()
    train_dataset = torch.load(data_loc(TRAIN))
    valid_dataset = torch.load(data_loc(VALID))
    print('loading done')
    # raise Exception("")

    _ = gat(get_data_sample())  # detect and init output model
    trainer = get_trainer(gat, data_loc(OUT), train_dataset, valid_dataset)
    trainer.data_collator = composite_data_collator  # to handle non tensor inputs without error

    check = get_latest_model()
    check = None if check is None else os.path.join(".", data_loc(OUT), check)
    print("checkpoint:", check)
    evaluate_model(gat, valid_dataset)
    trainer.train(model_path=check)
    trainer.save_model()

    evaluate_model(gat, valid_dataset)

