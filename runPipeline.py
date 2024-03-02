from src import logger
from src.pipeline.stage01 import DataIngestionTrainingPipeline
from src.pipeline.stage02 import DataCleaningPipeline
from src.pipeline.util import get_data_for_test
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
    data_cleaning = DataCleaningPipeline()
    data_cleaning.main(data=get_data_for_test())
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
