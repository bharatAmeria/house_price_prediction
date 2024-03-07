from src import logger
from src.components.feature_engineering import FeatureEngineeringConfig, OutlierTreatment, MissingValueImputation, \
    FeatureSelection, FeatureEngineering, FeatureEngineeringStrategy
from src.entity.config_entity import DataDividerConfig
from src.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage02_data_cleaning import CleaningStage


# STAGE_NAME = "Data Ingestion stage"
#
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e
#
STAGE_NAME = "Data Cleaning Stage"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    cleaning_stage = CleaningStage()
    cleaned_data = cleaning_stage.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Feature Engineering Stage"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} 1: FeatureEngineeringConfig started <<<<<<")

    # Instantiate FeatureEngineering
    fe_strategy = FeatureEngineeringStrategy()
    fe = FeatureEngineering(data=cleaned_data, strategy=fe_strategy)

    # Feature Engineering Stage 1: FeatureEngineeringConfig
    fe_config = FeatureEngineeringConfig()
    cleaned_data = fe_config.handle_FE(cleaned_data)

    logger.info(f">>>>>> stage {STAGE_NAME} 1: FeatureEngineeringConfig completed <<<<<<")
    logger.info(f">>>>>> stage {STAGE_NAME} 2: OutlierTreatment started <<<<<<")

    # Feature Engineering Stage 2: OutlierTreatment
    outlier_treatment = OutlierTreatment()
    cleaned_data = outlier_treatment.handle_FE(cleaned_data)

    logger.info(f">>>>>> stage {STAGE_NAME} 2: OutlierTreatment completed <<<<<<")
    logger.info(f">>>>>> stage {STAGE_NAME} 3: MissingValueImputation started <<<<<<")

    # Feature Engineering Stage 3: MissingValueImputation
    missing_value_imputation = MissingValueImputation()
    cleaned_data = missing_value_imputation.handle_FE(cleaned_data)

    logger.info(f">>>>>> stage {STAGE_NAME} 3: MissingValueImputation completed <<<<<<")
    logger.info(f">>>>>> stage {STAGE_NAME} 4: FeatureSelection started <<<<<<")

    # Feature Engineering Stage 4: FeatureSelection
    feature_selection = FeatureSelection()
    cleaned_data = feature_selection.handle_FE(cleaned_data)

    logger.info(f">>>>>> stage {STAGE_NAME} 4: FeatureSelection completed <<<<<<")
    logger.info(f">>>>>> stage {STAGE_NAME} 5: DataDivideStrategy started <<<<<<")

    # Feature Engineering Stage 5: DataDivideStrategy
    # data_divide_strategy = DataDivideStrategy()
    # train_data_path, test_data_path = data_divide_strategy.handle_FE(cleaned_data)

    logger.info(f">>>>>> stage {STAGE_NAME} 5: DataDivideStrategy completed <<<<<<")
    logger.info(f">>>>>> All Feature Engineering Stages completed successfully! <<<<<<")
    logger.info(f"*******************\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

