import os

import nlp
import torch
from transformers import LongformerTokenizerFast, Trainer, LongformerForQuestionAnswering, TrainingArguments

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


trainer.train(model_path=OUT)
trainer.save_model()

# model = model.cuda()
# model.eval()
