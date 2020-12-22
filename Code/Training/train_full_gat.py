import os
import sys

import torch
from transformers import WEIGHTS_NAME

from Code.Training import device

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Models.GNNs.ContextGNNs.context_gat_longformer_semi_output import ContextGATLongSemiOutput
from Code.Models.GNNs.ContextGNNs.context_gat_longformer_output import ContextGATLongOutput
from Code.Models.GNNs.TokenGNNs.gat_token_construction import GatTokenConstruction
from Code.Training.Utils.text_and_tensor_coalator import composite_data_collator
from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Training.Utils.eval_utils import evaluate_full_gat
from Code.Training.Utils.initialiser import get_trainer, get_span_composite_model, FEATURES
from Code.Training.Utils.dataset_utils import get_processed_data_sample, get_latest_model, process_gat_dataset, \
    load_processed_datasets, data_loc


from Code.Config import gec, gnnc
from Code.Config import gcc

"""how to name the preprocessed data files"""
TRAIN = 'train_data.pt'
VALID = 'valid_data.pt'
# MODEL_FOLDER = "context_model"
MODEL_FOLDER = "context_model_long_out"
# MODEL_FOLDER = "token_gat"


DATASET = "squad"  # "qangaroo"  # "squad"
VERSION = None  # "wikihop"
# DATASET = "qangaroo"  # "qangaroo"  # "squad"
# VERSION = "wikihop"


if __name__ == "__main__":
    """
        full meaning online text->embs
    """
    print("starting", MODEL_FOLDER, "model init")
    if MODEL_FOLDER == "token_gat":
        """
            uses the configurable graph embedding system, but with token only settings
            passes the graph embedding through a GNN, then through the huggingface span prediction output model
            currently incompatible with wikihop. Ie squad only
        """
        gat = get_span_composite_model(wrap_class=GatTokenConstruction)
    elif MODEL_FOLDER == "context_model":
        """
            the full configurable gnn. supports any node types and arbitrary connections
            uses a custom output model which either does span prediction or candidate selection
            
            currently does not work  -  been trying to get this one working
        """
        embedder = gec.get_graph_embedder(gcc)
        gat = ContextGAT(embedder, gnnc).to(device)
    else:
        embedder = gec.get_graph_embedder(gcc)
        gat = ContextGATLongSemiOutput(embedder, gnnc, FEATURES).to(device)

    print('loading data')
    process_gat_dataset(DATASET, VERSION, TRAIN, VALID)
    train_dataset, valid_dataset = load_processed_datasets(DATASET, VERSION, TRAIN, VALID)
    print('data loading done')

    _ = gat(get_processed_data_sample(DATASET, VERSION, TRAIN))  # detect and init output model
    print("inited model with:", sum(p.numel() for p in gat.parameters() if p.requires_grad), "trainable params")

    model_loc = data_loc(DATASET, VERSION, MODEL_FOLDER)
    trainer = get_trainer(gat, model_loc, train_dataset, valid_dataset)
    trainer.data_collator = composite_data_collator  # to handle non tensor inputs without error

    check = get_latest_model(DATASET, VERSION, MODEL_FOLDER)
    check = None if check is None else os.path.join(model_loc, check)
    if check is not None:
        gat.load_state_dict(torch.load(os.path.join(check, WEIGHTS_NAME)))
    print("training from checkpoint:", check)
    trainer.train(model_path=check)
    trainer.save_model()
    evaluate_full_gat(DATASET, VERSION, gat, valid_dataset)

