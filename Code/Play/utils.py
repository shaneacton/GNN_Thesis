import torch
from transformers import LongformerConfig, LongformerTokenizerFast, LongformerForQuestionAnswering, LongformerModel

from Code.Play.wrap import Wrap

device = torch.device("cpu")

FEATURES = 402
INTERMEDIATE_FEATURES = 600
HEADS = 6
ATTENTION_WINDOW = 128
LAYERS = 7

PRETRAINED = "valhalla/longformer-base-4096-finetuned-squadv1"

_tokenizer = None


@property
def tokenizer() -> LongformerTokenizerFast:
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

    configuration.vocab_size = tokenizer.vocab_size
    return configuration


def get_fresh_span_longformer():
    """no pretraining"""
    configuration = get_longformer_config()

    qa = LongformerForQuestionAnswering(configuration).to(device)

    return qa


def get_composit_qa_longformer(output_model):
    """frozen pretrained longformer base with a specialised trainable output longformer"""

    qa = LongformerModel.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

    qa = Wrap(qa, output_model)
    return qa