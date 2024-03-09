from runPipeline import cleaning_stage
from src import logger
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringStrategy, OutlierTreatment, \
    FeatureSelection, DataDivideStrategy, MissingValueImputation, FeatureEngineeringConfig
from src.components.model_trainer import ModelTrainer
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
        data_divide_strategy = DataDivideStrategy()
        X_train, y_train, X_test, y_test = data_divide_strategy.handle_FE(cleaned_data)

        # For example, you can print the shapes of the datasets:
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        model_trainer = ModelTrainer()
        r2_square = model_trainer.initiate_model_trainer(X_train, y_train)
        return r2_square


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
