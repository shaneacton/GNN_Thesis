import os
from typing import Iterable

from Code.Data.Text.data_sample import DataSample


class DataReader:

    def __init__(self, ):
        super().__init__()

    def get_data_samples(self, file_path: str) -> Iterable[DataSample]:
        raise NotImplementedError()

    @staticmethod
    def raw_data_location():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","Raw")