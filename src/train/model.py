from typing import Any

import pandas as pd


class Model:
    def __init__(self) -> None:
        pass

    def convert_input(self, input_data: pd.DataFrame) -> Any:
        """
            Convert the data to the correct input format for the model
            By default, we assume that it is already in the correct format
        :param input_data:
        :return:
        """
        return input_data

    def train(self, data_loaders: dict):
        pass

    def test(self, test_data: pd.DataFrame) -> (float, float, float):
        pass
