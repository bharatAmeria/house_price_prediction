import pandas as pd

from src import logger
from src.components.data_cleaning import DataCleaning, DataPreprocessStrategy
from src.components.data_ingestion import IngestData
from src.components.feature_engineering import FeatureEngineering, DataDivideStrategy
from src.config.configuration import ConfigurationManager


class Test:
    def __init__(self):
        pass

    def main(self):
        # config = ConfigurationManager()

        # data_ingestion_config = config.get_data_ingestion_config()

        # Instantiate IngestData
        # data_ingestion = IngestData(config=data_ingestion_config)

        # Download and extract data
        # data_ingestion.download_file()
        # data_ingestion.extract_zip_file()

        # Get data
        # raw_data = data_ingestion.get_data()

        raw_data = pd.read_csv('/Users/bharataameriya/Documents/projects/house_price_prediction/artifacts/data_ingestion/gurgaon_properties/gurgaon_properties.csv')

        data_cleaning = DataCleaning(data=raw_data, strategy=DataPreprocessStrategy())

        # Get cleaned data
        cleaned_data = data_cleaning.handle_data()

        # Instantiate FeatureEngineering
        feature_engineering = FeatureEngineering()

        # Perform feature engineering
        final_data = feature_engineering.handle_FE(cleaned_data)

        # Instantiate DataDivision
        data_division = DataDivideStrategy()

        # Divide the data into train and test sets
        X_train, X_test, y_train, y_test = data_division.handle_FE(final_data)


STAGE_NAME = "test"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = Test()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
