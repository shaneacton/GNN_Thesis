import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

import nlp
import torch
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering

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


def save_dataset():

    # load train and validation split of squad
    train_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.TRAIN, name=VERSION)
    valid_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.VALIDATION, name=VERSION)

    train_dataset = train_dataset.map(encoder.get_qa_features)
    valid_dataset = valid_dataset.map(encoder.get_qa_features) #, load_from_cache_file=False)

    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    torch.save(train_dataset, data_loc(TRAIN))
    torch.save(valid_dataset, data_loc(VALID))


def evaluate_model(model, valid_dataset):
    model = model.cuda()
    model.eval()

    dataloader = DataLoader(valid_dataset, batch_size=16)

    answers = []
    with torch.no_grad():
        for batch in nlp.tqdm(dataloader):
            start_scores, end_scores = model(input_ids=batch['input_ids'].cuda(),
                                             attention_mask=batch['attention_mask'].cuda(), return_dict=False)
            for i in range(start_scores.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                answer = ' '.join(all_tokens[torch.argmax(start_scores[i]): torch.argmax(end_scores[i]) + 1])
                ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
                answer = tokenizer.decode(ans_ids)
                answers.append(answer)

    predictions = []
    references = []
    valid_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.VALIDATION, name=VERSION)  # raw text version

    for ref, pred in zip(valid_dataset, answers):
        predictions.append(pred)
        # print("ref:", ref)
        references.append(ref['answers']['text'])

    print(evaluate(references, predictions))


save_dataset()

print("starting model init")
# model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
model = get_composite_span_longformer()

# Get datasets
print('loading data')
train_dataset = torch.load(data_loc(TRAIN))
valid_dataset = torch.load(data_loc(VALID))
print('loading done')

# evaluate_model(model, valid_dataset)

trainer = get_trainer(model, data_loc(OUT), train_dataset, valid_dataset)


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


check = get_latest_model()
check = None if check is None else os.path.join(".", data_loc(OUT), check)
print("checkpoint:", check)
trainer.train(model_path=check)
trainer.save_model()

evaluate_model(model, valid_dataset)
