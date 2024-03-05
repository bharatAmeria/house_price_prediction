from typing import Annotated

import mlflow
import pandas as pd
from click import Tuple
from sklearn.base import RegressorMixin

from src import logger
from src.components.evaluation import Evaluation, MSE, R2Score, RMSE
from src.config.configuration import ConfigurationManager

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
        config = ConfigurationManager()
        config.get_evaluation_config()
        Evaluation()
        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2_score, rmse

    # evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main(model=RegressorMixin, x_test=pd.DataFrame)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
