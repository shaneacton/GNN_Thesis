import pickle
from os.path import exists

import nlp
import torch
from tqdm import tqdm

from Code.HDE.hde_glove import HDEGloveStack
from Code.HDE.hde_long import HDELongStack
from Code.HDE.wikipoint import Wikipoint
from Code.Training import device
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset
from Viz.loss_visualiser import visualise_training_data


def get_processed_wikihop(save_path, glove_embedder, max_examples=-1, split=nlp.Split.TRAIN):
    global has_loaded
    save_path = "/".join(save_path.split("/")[:-1]) + "/"  # processed data can be used by all model variants
    suffix = split._name + ".data"
    data_path = save_path + suffix
    if exists(data_path):  # has been processed before
        print("loading preprocessed wikihop", split)
        filehandler = open(data_path, 'rb')
        data = pickle.load(filehandler)
        filehandler.close()
        return data

    print("loading wikihop unprocessed")
    train = list(load_unprocessed_dataset("qangaroo", "wikihop", split))
    train = train[:max_examples] if max_examples > 0 else train
    print("num examples:", len(train))

    print("processing wikihop")
    processed_examples = [Wikipoint(ex, glove_embedder=glove_embedder) for ex in tqdm(train)]
    save_training_data(processed_examples, save_path, suffix=suffix)
    return processed_examples


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(save_path, hidden_size=200, embedded_dims=100, optimizer_type="sgd", **model_kwargs):
    hde = None
    if exists(save_path):
        try:
            checkpoint = torch.load(save_path)
            hde = checkpoint["model"].to(device)
            optimizer = get_optimizer(hde, type=optimizer_type)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("loading checkpoint model", hde.name, "at:", save_path, "e:", hde.last_epoch, "i:", hde.last_example,
                  "with", num_params(hde), "trainable params")
        except Exception as e:
            print(e)
            print("cannot load model at", save_path)
    if hde is None:
        hde = HDEGloveStack(hidden_size=hidden_size, embedded_dims=embedded_dims, **model_kwargs).to(device)
        # hde = HDELongStack(hidden_size=hidden_size, **model_kwargs).to(device)

        optimizer = get_optimizer(hde, type=optimizer_type)
        print("inited model", hde.name, repr(hde), "with:", num_params(hde), "trainable params")

    return hde, optimizer


def get_training_data(save_path):
    if exists(save_path):
        filehandler = open(save_path + ".data", 'rb')
        data = pickle.load(filehandler)
        filehandler.close()
        return data

    return {"losses": [], "train_accs": [], "valid_accs": []}


def get_optimizer(model, type="sgd", lr=0.001):
    print("using", type, "optimiser")
    params = (p for p in model.parameters() if p.requires_grad)
    if type == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if type == "adamw":
            return torch.optim.AdamW(params, lr=lr)
    if type == "adam":
        return torch.optim.Adam(params, lr=lr)

    raise Exception("unreckognised optimizer arg: " + type)


def plot_training_data(data, save_path, print_loss_every, num_training_examples):
    path = save_path + "_losses.png"
    losses, train_accs, valid_accs = data["losses"], data["train_accs"], data["valid_accs"]

    num_prints = len(losses)
    num_trained_examples = num_prints * print_loss_every
    num_epochs = num_trained_examples / num_training_examples
    epochs = [num_epochs * i/len(losses) for i in range(len(losses))]

    visualise_training_data(losses, accuracies=train_accs, show=False, save_path=path, epochs=epochs, valid_accs=valid_accs)


def save_training_data(data, save_path, suffix=".data"):
    filehandler = open(save_path + suffix, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()


def save_config(cfg, save_path):
    filehandler = open(save_path + ".cfg", 'wb')
    pickle.dump(cfg, filehandler)
    filehandler.close()