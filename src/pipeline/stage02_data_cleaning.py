import pandas as pd

from src import logger
from src.components.data_cleaning import DataCleaning, DataPreprocessStrategy

STAGE_NAME = 'Data Cleaning Stage'

class CleaningStage:
    def __init__(self):
        pass

    @staticmethod
    def main():
        raw_data = pd.read_csv("artifacts/data_ingestion/gurgaon_properties/gurgaon_properties.csv")
        data_cleaning = DataCleaning(data=raw_data, strategy=DataPreprocessStrategy())

        cleaned_data = data_cleaning.handle_data()
        return cleaned_data


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = CleaningStage
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
