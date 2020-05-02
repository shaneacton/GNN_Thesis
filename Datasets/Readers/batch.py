from Code.GNN_Playground.Data.training_example import TrainingExample


class batch:

    """
        a batch is a collection of data_examples with
        utils to pad and combine vecs from these examples
    """

    def __init__(self, data_examples: TrainingExample):
        self.data_examples = data_examples

    def get