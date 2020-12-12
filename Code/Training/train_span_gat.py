import os
import sys


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Training.dataset_utils import get_latest_model, process_span_dataset, load_processed_datasets, data_loc
from Code.Models.GNNs.TokenGNNs.gat_comp_long_enc import GatWrapLongEnc
from Code.Models.GNNs.TokenGNNs.gat_composite import GatWrap
from Code.Models.GNNs.TokenGNNs.composite import Wrap
from Code.Training.eval_utils import evaluate_span_model
from Code.Play.initialiser import get_trainer, get_span_composite_model


WRAP_CLASS = GatWrapLongEnc

TRAIN = 'long_train_data.pt'
VALID = 'long_valid_data.pt'
model_name = "GAT" if WRAP_CLASS == GatWrap else "Lin" if WRAP_CLASS == Wrap else "GATEnc"
MODEL_FOLDER = model_name + "_models"

# DATASET = "squad"
# VERSION = None
DATASET = "qangaroo"
VERSION = "wikihop"


if __name__ == "__main__":
    print("starting model init")
    model = get_span_composite_model(wrap_class=WRAP_CLASS)

    # Get datasets
    print('loading data')
    process_span_dataset(DATASET, VERSION, TRAIN, VALID)
    train_dataset, valid_dataset = load_processed_datasets(DATASET, VERSION, TRAIN, VALID)

    print('loading done')
    model_loc = data_loc(DATASET, VERSION, MODEL_FOLDER)
    trainer = get_trainer(model, model_loc, train_dataset, valid_dataset)

    check = get_latest_model(DATASET, VERSION, MODEL_FOLDER)
    check = None if check is None else os.path.join("../Play", model_loc, check)
    print("training from checkpoint:", check)
    trainer.train(model_path=check)
    trainer.save_model()

    evaluate_span_model(DATASET, VERSION, model, valid_dataset)