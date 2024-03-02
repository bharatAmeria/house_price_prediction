from src import logger, config
from src.pipeline.stage01 import DataIngestionTrainingPipeline
from src.pipeline.stage02 import DataCleaningPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Cleaning stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_cleaning = DataCleaningPipeline(data_dir=config.data_dir)
    data_cleaning.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
