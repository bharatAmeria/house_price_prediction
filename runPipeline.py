from src import logger, config
from src.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage02_data_cleaning import CleaningStage
from src.pipeline.stage03_feature_engg import FeatureEngineeringStage
# from src.pipeline.stage_04_evaluation import EvaluationPipeline
from src.pipeline.stage_05_model_train import main

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Cleaning Stage"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    cleaning_stage = CleaningStage()
    cleaning_stage.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Feature Engineering Stage"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    cleaned_data = CleaningStage.main()
    feature_engg = FeatureEngineeringStage()
    feature_engg.main(cleaned_data)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation stage"

# try:
#     logger.info(f"*******************")
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     model_evaluation = EvaluationPipeline()
#     model_evaluation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Model Training"


try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = main(x_train=df, x_test=df, y_train=df, y_test=df, config=config)
    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


except Exception as e:
    logger.exception(e)
    raise e
