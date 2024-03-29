import time
from math import floor
from statistics import mean

from Code.Utils.eval_utils import get_acc_and_f1
from Config.config import conf
from Code.Utils.wandb_utils import wandb_run, use_wandb


class TrainingResults:

    def __init__(self):
        self.answers = []
        self.candidate_counts = []
        self.predictions = []

        self.all_losses = []
        self.mean_losses = []
        self.train_accs = []
        self.valid_accs = []
        self.epochs = []

        self.num_discarded = []
        self.num_edges = []
        self.num_nodes = []

    @property
    def epoch(self):
        return self.epochs[-1]

    def report_step(self, loss, prediction, answer, num_candidates):
        self.all_losses.append(loss)
        self.predictions.append(prediction)
        self.answers.append(answer)
        self.candidate_counts.append(num_candidates)

    def log_last_steps(self, epoch):
        self.epochs.append(epoch)
        mean_loss = mean(self.all_losses[-conf.print_loss_every:-1])
        training_acc = get_acc_and_f1(self.answers[-conf.print_loss_every:-1],
                                      self.predictions[-conf.print_loss_every:-1])['exact_match']
        self.train_accs.append(training_acc)
        chances = [1.0/count for count in self.candidate_counts[-conf.print_loss_every:-1]]
        chance = mean(chances)
        chance_fac = training_acc/chance
        print("e", epoch, "loss:", mean_loss, "acc:", training_acc, "chance:", chance, "chance_fac:", chance_fac)
        if use_wandb:
            wandb_run().log({"loss": mean_loss, "train_acc": training_acc, "epoch": epoch,
                         "chance": chance, "chance_fac": chance_fac })

    def log_epoch(self, epoch, valid_acc, num_discarded_examples, epoch_start_time, num_fastforward_examples):
        self.num_discarded.append(num_discarded_examples)
        self.valid_accs.append(valid_acc)
        start, end = self.get_epoch_span(epoch)
        epoch_time = time.time() - epoch_start_time
        skipped_frac = num_fastforward_examples / (num_fastforward_examples + (end-start) * conf.print_loss_every)
        adjusted_epoch_time = epoch_time / (1-skipped_frac)

        epoch_train_acc = get_acc_and_f1(self.answers[start: end], self.predictions[start: end])['exact_match']

        print("e", epoch, "completed. Training acc:", epoch_train_acc, "valid_acc:", valid_acc, "num discarded:",
              num_discarded_examples, "time:", adjusted_epoch_time)

        if use_wandb:
            wandb_run().log({"valid_acc": valid_acc, "epoch": epoch + 1, "epoch_time": adjusted_epoch_time,
                             "num_discarded_examples": num_discarded_examples})

    def get_epoch_span(self, epoch):
        start, end, current_epoch = -1, -1, -1
        for i, e in enumerate(self.epochs):
            round_e = floor(e)
            if round_e > current_epoch:
                "found new epoch index"
                if round_e == epoch:
                    start = i
                if round_e == epoch + 1:
                    end = i
        return start, end
