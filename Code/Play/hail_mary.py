import os
from os.path import isfile, join

import sys

import nlp
import torch
from transformers import LongformerTokenizerFast, Trainer, LongformerForQuestionAnswering, TrainingArguments

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Play.encoding import TextEncoder

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
encoder = TextEncoder(tokenizer)

TRAIN = 'train_data.pt'
VALID = 'valid_data.pt'
OUT = "out"
MODEL = "model"

DATASET = "squad"  # "qangaroo"  # "squad"
VERSION = None  # "wikihop"


def save_dataset():

    # load train and validation split of squad
    train_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.TRAIN, name=VERSION)
    valid_dataset = nlp.load_dataset(path=DATASET, split=nlp.Split.VALIDATION, name=VERSION)

    train_dataset = train_dataset.map(encoder.get_span_features)
    valid_dataset = valid_dataset.map(encoder.get_span_features) #, load_from_cache_file=False)

    # set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    torch.save(train_dataset, TRAIN)
    torch.save(valid_dataset, VALID)


# save_dataset()

print("starting model init")
model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

# Get datasets
print('loading data')
train_dataset = torch.load(TRAIN)
valid_dataset = torch.load(VALID)
print('loading done')

train_args = TrainingArguments(OUT)
train_args.per_device_train_batch_size = 1
train_args.do_eval = True
train_args.evaluation_strategy = "steps"
train_args.eval_steps = 2000
train_args.do_train = True

train_args.save_steps = 500
train_args.overwrite_output_dir = True
train_args.save_total_limit = 2


# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    prediction_loss_only=True,
)


def get_latest_model():
    checks = [f for f in os.listdir(OUT) if isfile(join(OUT, f))]
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


trainer.train(model_path=get_latest_model())
trainer.save_model()

# model = model.cuda()
# model.eval()
