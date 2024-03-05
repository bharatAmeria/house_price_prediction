import mlflow
import pandas as pd
from sklearn.base import RegressorMixin

from src import logger
from src.components.model_devlopment import RandomForestModel, XGBoostModel, LinearRegressionModel, HyperparameterTuner
from src.entity.config_entity import ModelNameConfig

STAGE_NAME = 'Model trainer'


def main(x_train: pd.DataFrame,
         x_test: pd.DataFrame,
         y_train: pd.Series,
         y_test: pd.Series,
         config: ModelNameConfig,
         ) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
        :param y_test:
        :param y_train:
        :param x_test:
        :param x_train:
        :param config:
    """
    try:

        if config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logger.error(e)
        raise e


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = main(x_train=pd.DataFrame,
                   x_test=pd.DataFrame,
                   y_train=pd.Series,
                   y_test=pd.Series,
                   config=ModelNameConfig)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
