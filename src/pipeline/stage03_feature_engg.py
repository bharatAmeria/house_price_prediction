from src import logger
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringStrategy, OutlierTreatment, \
    FeatureSelection, DataDivideStrategy

STAGE_NAME = 'Feature Engineering Stage'

class FeatureEngineeringStage:
    def __init__(self):
        pass

    @staticmethod
    def main(cleaned_data):
        # Instantiate FeatureEngineering
        fe = FeatureEngineering(cleaned_data, strategy=FeatureEngineeringStrategy)
        fe.handle_FE()

        outlier = OutlierTreatment()
        outlier_data = outlier.handle_FE(df=cleaned_data)

        feature_selection = FeatureSelection()
        feature_selection_data = feature_selection.handle_FE(outlier_data)

        # Instantiate DataDivision
        data_division = DataDivideStrategy()

        # # Divide the data into train and test sets
        X_train, X_test, y_train, y_test = data_division.handle_FE(feature_selection_data)


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureEngineeringStage
        obj.main(cleaned_data=df)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
