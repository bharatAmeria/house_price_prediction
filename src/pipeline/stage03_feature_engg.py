from src import logger
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringStrategy, OutlierTreatment, \
    FeatureSelection, DataDivideStrategy
from src.pipeline.stage02_data_cleaning import CleaningStage

STAGE_NAME = 'Feature Engineering Stage'

class FeatureEngineeringStage:
    def __init__(self):
        pass

    @staticmethod
    def main(cleaned_data):
        # Instantiate FeatureEngineering
        fe_strategy = FeatureEngineeringStrategy()
        fe = FeatureEngineering(data=cleaned_data, strategy=fe_strategy)
        fe.handle_FE()

        outlier_strategy = OutlierTreatment()
        outlier = FeatureEngineering(data=cleaned_data, strategy=outlier_strategy)
        outlier_data = outlier.handle_FE()

        feature_selection = FeatureSelection()
        feature_selection_data = feature_selection.handle_FE(outlier_data)

        # Instantiate DataDivision
        data_division = DataDivideStrategy()

        # # Divide the data into train and test sets
        X_train, X_test, y_train, y_test = data_division.handle_FE(feature_selection_data)

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        cleaned_data = CleaningStage.main()
        obj = FeatureEngineeringStage()
        obj.main(cleaned_data)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
