import torch
from transformers import LongformerConfig, LongformerTokenizerFast, LongformerModel, \
    TrainingArguments, Trainer, T5Config

from Code.Training import dev

FEATURES = 402
INTERMEDIATE_FEATURES = 600
HEADS = 6
ATTENTION_WINDOW = 128

BATCH_SIZE = 1
NUM_EPOCHS = 2

# PRETRAINED = "valhalla/longformer-base-4096-finetuned-squadv1"
PRETRAINED = "allenai/longformer-base-4096"


_tokenizer = None


def get_trainer(model, outdir, train_dataset, valid_dataset):
    train_args = TrainingArguments(outdir)
    train_args.per_device_train_batch_size = BATCH_SIZE
    train_args.do_eval = False
    train_args.evaluation_strategy = "no"
    # train_args.eval_steps = 2000
    train_args.do_train = True                                                                                                                                                                                                                                                                                                                                                                              
    train_args.no_cuda = dev == torch.device("cpu")

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
