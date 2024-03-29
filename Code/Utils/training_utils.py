import pickle
from os.path import exists

from torch.optim.lr_scheduler import LambdaLR

from Checkpoint.checkpoint_utils import training_results_path, save_binary_data
from Code.Training.training_results import TrainingResults


def get_training_results(name):
    path = training_results_path(name)
    if exists(path):
        filehandler = open(path, 'rb')
        data = pickle.load(filehandler)
        filehandler.close()
        return data

    return TrainingResults()


# def plot_training_data(training_results, name, print_loss_every, num_training_examples):
#     visualise_training_data(losses, train_accs, epochs, name, show=False, valid_accs=valid_accs)


def save_training_results(data, name):
    save_binary_data(data, training_results_path(name))


def get_exponential_schedule_with_warmup(optimizer, num_grace_epochs=1, decay_fac=0.9):
    """roughly halves the lr every 7 epochs. at e 50, lr is 200 times lower"""

    def lr_lambda(epoch: float):
        if epoch <= num_grace_epochs:
            return 1
        t = epoch - num_grace_epochs
        # print("e:", epoch, "lr_f:", decay_fac ** t)

        return decay_fac ** t

    return LambdaLR(optimizer, lr_lambda)