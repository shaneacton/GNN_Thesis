import os
import sys


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Training.Utils.dataset_utils import get_latest_model, process_span_dataset, load_processed_datasets, data_loc
from Code.Models.GNNs.TokenGNNs.gat_comp_long_enc import GatWrapLongEnc
from Code.Models.GNNs.TokenGNNs.gat_composite import GatWrap
from Code.Models.GNNs.TokenGNNs.composite import Wrap
from Code.Training.Utils.eval_utils import evaluate_span_model
from Code.Training.Utils.initialiser import get_trainer, get_span_composite_model

"""
    a wrap class takes a pretrained longformer, and an untrained longformer output model for span prediction
    it sandwiches a GNN inbetween these two longformers
    different wrap classes tested different pipeline elements, all have worked so far
"""
WRAP_CLASS = GatWrapLongEnc

"""how to name the preprocessed data files"""
TRAIN = 'long_train_data.pt'
VALID = 'long_valid_data.pt'
model_name = "GAT" if WRAP_CLASS == GatWrap else "Lin" if WRAP_CLASS == Wrap else "GATEnc"
MODEL_FOLDER = model_name + "_models"

DATASET = "squad"
VERSION = None
# DATASET = "qangaroo"
# VERSION = "wikihop"


if __name__ == "__main__":
    """
        data is preprocessed into token ids, and text is dropped
        this approach is more inline with Longformer best practices
    """
    print("starting token span model init")
    model = get_span_composite_model(wrap_class=WRAP_CLASS)

    # Get datasets
    print('loading data')
    process_span_dataset(DATASET, VERSION, TRAIN, VALID)
    train_dataset, valid_dataset = load_processed_datasets(DATASET, VERSION, TRAIN, VALID)

    print('loading done')
    model_loc = data_loc(DATASET, VERSION, MODEL_FOLDER)
    trainer = get_trainer(model, model_loc, train_dataset, valid_dataset)

    check = get_latest_model(DATASET, VERSION, MODEL_FOLDER)
    check = None if check is None else os.path.join(model_loc, check)
    print("training from checkpoint:", check)
    trainer.train(model_path=check)
    trainer.save_model()

    evaluate_span_model(DATASET, VERSION, model, valid_dataset)