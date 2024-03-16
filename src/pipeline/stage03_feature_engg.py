from runPipeline import cleaning_stage
from src import logger
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringStrategy, OutlierTreatment, \
    FeatureSelection, DataDivideStrategy, MissingValueImputation, FeatureEngineeringConfig
from src.pipeline.stage02_data_cleaning import CleaningStage

STAGE_NAME = 'Feature Engineering Stage'


class FeatureEngineeringStage:
    def __init__(self):
        pass

    @staticmethod
    def main(cleaned_data):
        # Cleaning Stage 1: CleaningConfig
        cleaned_data = cleaning_stage.main()

        # Instantiate FeatureEngineering
        fe_strategy = FeatureEngineeringStrategy()
        fe = FeatureEngineering(data=cleaned_data, strategy=fe_strategy)

        # Feature Engineering Stage 1: FeatureEngineeringConfig
        fe_config = FeatureEngineeringConfig()
        cleaned_data = fe_config.handle_FE(cleaned_data)

        # Feature Engineering Stage 2: OutlierTreatment
        outlier_treatment = OutlierTreatment()
        cleaned_data = outlier_treatment.handle_FE(cleaned_data)

        # Feature Engineering Stage 3: MissingValueImputation
        missing_value_imputation = MissingValueImputation()
        cleaned_data = missing_value_imputation.handle_FE(cleaned_data)

        # Feature Engineering Stage 4: FeatureSelection
        feature_selection = FeatureSelection()
        cleaned_data = feature_selection.handle_FE(cleaned_data)

        return cleaned_data

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
