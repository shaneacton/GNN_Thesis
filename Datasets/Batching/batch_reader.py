from Code.Config import eval_conf
from Datasets.Batching.samplebatch import SampleBatch
from Datasets.Batching.batch_item import BatchItem
from Datasets.Readers.data_reader import DataReader


class BatchReader:

    """
        some datasets present multiple questions per context
        these multiple questions are wrapped up in a single data_example
        the batch reader takes in a data reader, and wraps its data_example iterator
        with a batch_iterator which collects multiple context,question pairs and groups them
    """

    def __init__(self, data_reader: DataReader, batch_size, data_path):
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.data_path = data_path

        self._length = -1  # must be calculated by looping

    @property
    def dataset(self):
        return self.data_reader.datset_name

    def __len__(self):
        if self._length == -1:  # only calculate once
            i = 0
            for _ in self.get_all_batches():  # full set
                i += 1
            self._length = i
        return self._length

    def get_train_batches(self, test_offset=0):
        return self.get_batches_subset(test_offset, False)

    def get_test_batches(self, test_offset=0):
        return self.get_batches_subset(test_offset, True)

    def get_batches_subset(self, test_offset, test: bool):
        """
        can be used for cross validation by stepping through test offset
        """
        num_batches = len(self)
        i = 0
        for batch in self.get_all_batches():
            frac = i / num_batches

            is_in_test_section = test_offset <= frac < test_offset + eval_conf.test_set_frac
            if is_in_test_section and test:
                yield batch

            if not is_in_test_section and not test:
                yield batch  # is train section

            i += 1

    def get_all_batches(self):
        batch = SampleBatch(self.batch_size)

        i = 0
        for data_example in self.data_reader.get_data_samples(self.data_path):
            for question in data_example.questions:
                batch.add_batch_item(BatchItem(data_example, question))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = SampleBatch(self.batch_size)
                    i += 1