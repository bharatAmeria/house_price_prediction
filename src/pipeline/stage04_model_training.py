from runPipeline import cleaned_data
from src import logger
from src.components.feature_engineering import DataDivideStrategy
from src.components.modelTraining import Model
from src.constants.constants import ModelNameConfig

STAGE_NAME = 'Model Training Stage'


class ModelTrainingStage:
    def __init__(self):
        pass

    @staticmethod
    def main():
        # Feature Engineering Stage 5: DataDivideStrategy
        data_divide_strategy = DataDivideStrategy()
        X_train, y_train, X_test, y_test = data_divide_strategy.handle_FE(cleaned_data)

        # training stage
        model = Model()
        model = model.main(X_train, y_train, X_test, y_test, ModelNameConfig)
        return model


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        training = ModelTrainingStage.main()
        obj = ModelTrainingStage
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
