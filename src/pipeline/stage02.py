from typing import Annotated, Tuple

import pandas as pd

from src import logger
from src.components.data_cleaning import DataPreprocessStrategy, DataCleaning, DataDivideStrategy


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def clean_data(data: pd.DataFrame, ) -> Tuple[
        Annotated[pd.DataFrame, "x_train"],
        Annotated[pd.DataFrame, "x_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """Data cleaning class which preprocesses the data and divides it into train and test data.

        Args:
            data: pd.DataFrame
        """
        try:
            preprocess_strategy = DataPreprocessStrategy()
            data_cleaning = DataCleaning(data, preprocess_strategy)
            preprocessed_data = data_cleaning.handle_data()

            divide_strategy = DataDivideStrategy()
            data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
            x_train, x_test, y_train, y_test = data_cleaning.handle_data()
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logger.error(e)
            raise e


if __name__ == "__main__":
    pass
