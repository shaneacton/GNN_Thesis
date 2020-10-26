import os
from os.path import isfile, join

import sys

import nlp
import torch
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast, Trainer, LongformerForQuestionAnswering, TrainingArguments

from Code.Training.eval_utils import evaluate

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
train_args.per_device_train_batch_size = 8
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
    out = os.path.join(".", OUT)
    checks = os.listdir(out)
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
check = None if check is None else os.path.join(".", OUT, check)
print("checkpoint:", check)
trainer.train(model_path=check)
trainer.save_model()

model = model.cuda()
model.eval()

dataloader = DataLoader(valid_dataset, batch_size=16)

answers = []
with torch.no_grad():
    for batch in nlp.tqdm(dataloader):
        start_scores, end_scores = model(input_ids=batch['input_ids'].cuda(),
                                      attention_mask=batch['attention_mask'].cuda())
        for i in range(start_scores.shape[0]):
            all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
            answer = ' '.join(all_tokens[torch.argmax(start_scores[i]) : torch.argmax(end_scores[i])+1])
            ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
            answer = tokenizer.decode(ans_ids)
            answers.append(answer)


predictions = []
references = []
for ref, pred in zip(valid_dataset, answers):
    predictions.append(pred)
    references.append(ref['answers']['text'])

evaluate(references, predictions)