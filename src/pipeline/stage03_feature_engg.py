from runPipeline import cleaning_stage
from src import logger
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringStrategy, OutlierTreatment, \
    FeatureSelection, DataDivideStrategy, MissingValueImputation, FeatureEngineeringConfig
from src.entity.config_entity import DataDividerConfig
from src.pipeline.stage02_data_cleaning import CleaningStage

STAGE_NAME = 'Feature Engineering Stage'

class FeatureEngineeringStage:
    def __init__(self):
        pass

    def main(self, cleaned_data):

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

        # Feature Engineering Stage 5: DataDivideStrategy
        data_divider_config = DataDividerConfig()
        data_divide_strategy = DataDivideStrategy(config=data_divider_config)
        train_data_path, test_data_path = data_divide_strategy.handle_FE(cleaned_data)

        # # Instantiate FeatureEngineering
        # fe_strategy = FeatureEngineeringStrategy()
        # fe = FeatureEngineering(data=cleaned_data, strategy=fe_strategy)
        # fe_data = fe.handle_FE()
        # return fe_data

        # outlier_strategy = OutlierTreatment()
        # outlier = FeatureEngineering(data=cleaned_data, strategy=outlier_strategy)
        # outlier_data = outlier.handle_FE()
        #
        # missing_strategy = MissingValueImputation()
        # missing = missing_strategy.handle_FE(data=outlier_data)
        #
        # feature_selection = FeatureSelection()
        # feature_selection_data = feature_selection.handle_FE(data=missing)


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
