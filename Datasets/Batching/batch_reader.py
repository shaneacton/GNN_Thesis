from Code.GNN_Playground.Training import batch_size
from Datasets.Batching.batch import Batch
from Datasets.Batching.batch_item import BatchItem
from Datasets.Readers.data_reader import DataReader


class BatchReader:

    """
        some datasets present multiple questions per context
        these multiple questions are wrapped up in a single data_example
        the batch reader takes in a data reader, and wraps its data_example iterator
        with a batch_iterator which collects multiple context,question pairs and groups them
    """

    def __init__(self, data_reader: DataReader):
        self.data_reader = data_reader

    def get_batches(self, file_path):
        batch = Batch()
        for data_example in self.data_reader.get_training_examples(file_path):
            for question in data_example.questions:
                batch.add_batch_item(BatchItem(data_example, question))
                if len(batch) == batch_size:
                    yield batch
                    batch = Batch()