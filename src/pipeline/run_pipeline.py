from src import logger
from src.components.data_cleaning import DataCleaning, DataPreprocessStrategy
from src.components.data_ingestion import IngestData
from src.components.feature_engineering import FeatureEngineering, \
    OutlierTreatment, \
    FeatureSelection, \
    DataDivideStrategy
from src.components.model_devlopment import Model
from src.config.configuration import ConfigurationManager


class Test:
    def __init__(self):
        pass

    def main(self):
        # Instantiate IngestData
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = IngestData(config=data_ingestion_config)

        # Get data
        raw_data = data_ingestion.get_data()

        data_cleaning = DataCleaning(data=raw_data, strategy=DataPreprocessStrategy())

        # Get cleaned data
        cleaned_data = data_cleaning.handle_data()

        # Instantiate FeatureEngineering
        feature_engineering = FeatureEngineering()

        # Perform feature engineering
        final_data = feature_engineering.handle_FE(cleaned_data)

        outlier = OutlierTreatment()
        outlier_data = outlier.handle_FE(final_data)

        feature_selection = FeatureSelection()
        feature_selection_data = feature_selection.handle_FE(outlier_data)

        # Instantiate DataDivision
        data_division = DataDivideStrategy()

        # # Divide the data into train and test sets
        X_train, X_test, y_train, y_test = data_division.handle_FE(feature_selection_data)

        # Train the model
        model_train = Model()
        model_train.train(x_train=X_train, y_train=y_train)

        # Instantiate Evaluation
        score = model_train.optimize(trial=None, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
        print(score)


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
