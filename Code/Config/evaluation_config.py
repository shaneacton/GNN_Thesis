from Code.Config.config import Config


class EvaluationConfig(Config):

    def __init__(self):
        super().__init__()
        self.batch_size = 1
        self.num_epochs = 100

        self.learning_rate_base = 1.5e-2
        self.test_set_frac = 0.1

        self.max_train_batches = 1001
        self.max_test_batches = 200

        self.print_stats_every_n_samples = 100

