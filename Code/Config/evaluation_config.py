from Code.Config.config import Config


class EvaluationConfig(Config):

    def __init__(self):
        super().__init__()
        self.learning_rate_base = 1e-2
        self.test_set_frac = 0.1

        self.max_train_batches = 1001
        self.max_test_batches = 100

        self.print_batch_every = 100