import pickle
from os.path import exists

from torch.optim.lr_scheduler import LambdaLR

from Checkpoint.checkpoint_utils import training_results_path, save_binary_data
from Code.Training.training_results import TrainingResults


def get_training_results(name, backup=False):
    path = training_results_path(name, backup=backup)
    if exists(path):
        try:
            filehandler = open(path, 'rb')
            data = pickle.load(filehandler)
            filehandler.close()
        except Exception as e:
            print("training results file load error for", name, e)
            if not backup:
                return get_training_results(name, backup=True)
            raise e
        return data

    return TrainingResults()


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