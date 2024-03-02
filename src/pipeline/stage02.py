import pandas as pd
from src.pipeline.util import get_data_for_test
from src import logger
from src.components.data_cleaning import DataCleaning, DataPreprocessStrategy

STAGE_NAME = "Data Cleaning stage"


class DataCleaningPipeline:
    def __init__(self):
        pass

    def main(self, data: pd.DataFrame) -> None:
        """Data cleaning class which preprocesses the data and divides it into train and test data.

        Args:
            data: pd.DataFrame
        """
        try:
            preprocess_strategy = DataPreprocessStrategy()
            data_cleaning = DataCleaning(data, preprocess_strategy)
            preprocessed_data = data_cleaning.handle_data()

            logger.info(f"Data cleaning completed successfully")
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.error(f"Error in cleaning Data: {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataCleaningPipeline()
        obj.main(data=get_data_for_test())
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
