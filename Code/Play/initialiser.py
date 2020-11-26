from typing import Type

import torch
from transformers import LongformerConfig, LongformerTokenizerFast, LongformerForQuestionAnswering, LongformerModel, \
    TrainingArguments, Trainer, LongformerForMultipleChoice

from Code.Play.composite import Wrap
from Code.Play.gat_composite import GatWrap
from Code.Training import device

FEATURES = 402
INTERMEDIATE_FEATURES = 600
HEADS = 6
ATTENTION_WINDOW = 128
LAYERS = 1

BATCH_SIZE = 1
NUM_EPOCHS = 1

# PRETRAINED = "valhalla/longformer-base-4096-finetuned-squadv1"
PRETRAINED = "allenai/longformer-base-4096"


_tokenizer = None


def get_tokenizer() -> LongformerTokenizerFast:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = LongformerTokenizerFast.from_pretrained(PRETRAINED)
    return _tokenizer


def get_longformer_config():
    configuration = LongformerConfig()

    configuration.attention_window = ATTENTION_WINDOW
    configuration.hidden_size = FEATURES
    configuration.intermediate_size = INTERMEDIATE_FEATURES
    configuration.num_labels = 2
    configuration.max_position_embeddings = 4000
    configuration.type_vocab_size = 3
    configuration.num_attention_heads = HEADS
    configuration.num_hidden_layers = LAYERS

    configuration.vocab_size = get_tokenizer().vocab_size
    return configuration


def get_pretrained_longformer():
    return LongformerModel.from_pretrained(PRETRAINED)


def get_pretrained_tokeniser():
    return LongformerTokenizerFast.from_pretrained(PRETRAINED)


def get_fresh_span_longformer():
    """no pretraining"""
    configuration = get_longformer_config()
    qa = LongformerForQuestionAnswering(configuration).to(device)

    return qa


def get_composit_qa_longformer(output_model, wrap_class: Type):
    """frozen pretrained longformer base with a specialised trainable output longformer"""
    qa = wrap_class(get_pretrained_longformer(), output_model)
    return qa.to(device)


def get_span_composite_model(wrap_class: Type = Wrap):
    return get_composit_qa_longformer(get_fresh_span_longformer(), wrap_class)


def get_trainer(model, outdir, train_dataset, valid_dataset):
    train_args = TrainingArguments(outdir)
    train_args.per_device_train_batch_size = BATCH_SIZE
    train_args.do_eval = False
    train_args.evaluation_strategy = "no"
    # train_args.eval_steps = 2000
    train_args.do_train = True

    train_args.save_steps = 250
    train_args.overwrite_output_dir = True
    train_args.save_total_limit = 2
    train_args.num_train_epochs = NUM_EPOCHS

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        prediction_loss_only=True,
    )
    return trainer