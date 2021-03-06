import pickle
from os.path import exists

from torch.optim.lr_scheduler import LambdaLR

from Viz.loss_visualiser import visualise_training_data, get_continuous_epochs


def get_training_results(save_path):
    if exists(save_path):
        filehandler = open(save_path + ".data", 'rb')
        data = pickle.load(filehandler)
        filehandler.close()
        return data

    return {"losses": [], "train_accs": [], "valid_accs": []}


def plot_training_data(data, save_path, print_loss_every, num_training_examples):
    path = save_path + "_losses.png"
    losses, train_accs, valid_accs = data["losses"], data["train_accs"], data["valid_accs"]
    epochs = get_continuous_epochs(losses, num_training_examples, print_loss_every)
    # print("got epochs:", epochs)
    visualise_training_data(losses, train_accs, epochs, show=False, save_path=path, valid_accs=valid_accs)


def save_data(data, save_path, suffix=".data"):
    filehandler = open(save_path + suffix, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()


def save_conf(cfg, save_path):
    filehandler = open(save_path + ".cfg", 'wb')
    pickle.dump(cfg, filehandler)
    filehandler.close()


def get_exponential_schedule_with_warmup(optimizer, num_grace_epochs=1, decay_fac=0.9):
    """roughly halves the lr every 7 epochs. at e 50, lr is 200 times lower"""

    def lr_lambda(epoch: float):
        if epoch <= num_grace_epochs:
            return 1
        t = epoch - num_grace_epochs
        # print("e:", epoch, "lr_f:", decay_fac ** t)

        return decay_fac ** t

    return LambdaLR(optimizer, lr_lambda)