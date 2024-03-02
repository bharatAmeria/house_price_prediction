from typing import Annotated, Tuple

import pandas as pd
from src import logger, config
from src.components.data_cleaning import DataPreprocessStrategy, DataCleaning, DataDivideStrategy

STAGE_NAME = "Data Cleaning stage"

class DataCleaningPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def main(self) -> Tuple[
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
            data = self.data_dir
            data_cleaning = DataPreprocessStrategy()
            data_cleaning = DataCleaning(data_cleaning, data)
            preprocessed_data = data_cleaning.handle_data()

            divide_strategy = DataDivideStrategy()
            data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
            x_train, x_test, y_train, y_test = data_cleaning.handle_data()
            return x_train, x_test, y_train, y_test

        except Exception as e:
            logger.error(e)
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataCleaningPipeline(config.data_dir)
        x_train, x_test, y_train, y_test = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
