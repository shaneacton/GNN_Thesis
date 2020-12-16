import os
import sys

from Code.Models.GNNs.TokenGNNs.gat_token_construction import GatTokenConstruction

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))


from Code.Training.Utils.text_and_tensor_coalator import composite_data_collator
from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Training.Utils.eval_utils import evaluate_full_gat
from Code.Training.Utils.initialiser import get_trainer, get_span_composite_model
from Code.Training.Utils.dataset_utils import get_processed_data_sample, get_latest_model, process_gat_dataset, \
    load_processed_datasets, data_loc


from Code.Config import gec, gnnc
from Code.Config import gcc


TRAIN = 'train_data.pt'
VALID = 'valid_data.pt'
# MODEL_FOLDER = "context_model"
MODEL_FOLDER = "token_gat"


DATASET = "squad"  # "qangaroo"  # "squad"
VERSION = None  # "wikihop"
# DATASET = "qangaroo"  # "qangaroo"  # "squad"
# VERSION = "wikihop"


if __name__ == "__main__":
    print("starting gat model init")
    embedder = gec.get_graph_embedder(gcc)
    if MODEL_FOLDER == "token_gat":
        gat = get_span_composite_model(wrap_class=GatTokenConstruction)
    else:
        gat = ContextGAT(embedder, gnnc)

    print('loading data')
    process_gat_dataset(DATASET, VERSION, TRAIN, VALID)
    train_dataset, valid_dataset = load_processed_datasets(DATASET, VERSION, TRAIN, VALID)
    print('loading done')

    _ = gat(get_processed_data_sample(DATASET, VERSION, TRAIN))  # detect and init output model
    model_loc = data_loc(DATASET, VERSION, MODEL_FOLDER)
    trainer = get_trainer(gat, model_loc, train_dataset, valid_dataset)
    trainer.data_collator = composite_data_collator  # to handle non tensor inputs without error

    check = get_latest_model(DATASET, VERSION, MODEL_FOLDER)
    check = None if check is None else os.path.join("../Play", model_loc, check)
    print("training from checkpoint:", check)
    trainer.train(model_path=check)
    trainer.save_model()

    evaluate_full_gat(DATASET, VERSION, gat, valid_dataset)

