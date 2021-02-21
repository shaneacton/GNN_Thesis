from typing import Type

import torch
from transformers import LongformerConfig, LongformerTokenizerFast, LongformerForQuestionAnswering, LongformerModel, \
    TrainingArguments, Trainer, T5Config

from Code.Training import device

FEATURES = 402
INTERMEDIATE_FEATURES = 600
HEADS = 6
ATTENTION_WINDOW = 128

BATCH_SIZE = 1
NUM_EPOCHS = 2

# PRETRAINED = "valhalla/longformer-base-4096-finetuned-squadv1"
PRETRAINED = "allenai/longformer-base-4096"


_tokenizer = None


def get_tokenizer() -> LongformerTokenizerFast:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = LongformerTokenizerFast.from_pretrained(PRETRAINED)
    return _tokenizer


def get_longformer_config(hidden_size=FEATURES, num_layers=1, num_types=3):
    configuration = LongformerConfig()

    configuration.attention_window = ATTENTION_WINDOW
    configuration.hidden_size = hidden_size
    configuration.intermediate_size = INTERMEDIATE_FEATURES
    configuration.num_labels = 2
    configuration.max_position_embeddings = 4098
    configuration.type_vocab_size = num_types
    configuration.num_attention_heads = HEADS
    configuration.num_hidden_layers = num_layers

    configuration.vocab_size = get_tokenizer().vocab_size
    return configuration


def get_transformer_config(hidden_size=FEATURES, num_layers=1, num_types=3):
    configuration = T5Config()

    configuration.d_model = hidden_size
    configuration.d_ff = hidden_size * 4
    configuration.num_heads = HEADS
    configuration.num_layers = num_layers

    configuration.vocab_size = get_tokenizer().vocab_size
    return configuration


def get_pretrained_longformer():
    pret = LongformerModel.from_pretrained(PRETRAINED)
    for param in pret.parameters():
        param.requires_grad = False
    print("loading pretrained with:", sum(p.numel() for p in pret.parameters()), "params")
    return pret.to(device)


def get_trainer(model, outdir, train_dataset, valid_dataset):
    train_args = TrainingArguments(outdir)
    train_args.per_device_train_batch_size = BATCH_SIZE
    train_args.do_eval = False
    train_args.evaluation_strategy = "no"
    # train_args.eval_steps = 2000
    train_args.do_train = True                                                                                                                                                                                                                                                                                                                                                                              
    train_args.no_cuda = device == torch.device("cpu")

    train_args.save_steps = 250
    train_args.overwrite_output_dir = True
    train_args.save_total_limit = 2
    train_args.num_train_epochs = NUM_EPOCHS
    train_args.prediction_loss_only = True
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()
    return trainer
