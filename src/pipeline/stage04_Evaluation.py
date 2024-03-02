from typing import Annotated
import mlflow
import pandas as pd
from click import Tuple
from sklearn.base import RegressorMixin
from src.components.evaluation import MSE, RMSE, R2Score
from src import logger


class EvaluationPipeline:
    def __init__(self):
        pass

    def evaluation(self, model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"]]:
        """
        Args:
            model: RegressorMixin
            x_test: pd.DataFrame
            y_test: pd.Series
        Returns:
            r2_score: float
            rmse: float
        """
        try:
            # prediction = model.predict(x_test)
            # evaluation = Evaluation()
            # r2_score = evaluation.r2_score(y_test, prediction)
            # mlflow.log_metric("r2_score", r2_score)
            # mse = evaluation.mean_squared_error(y_test, prediction)
            # mlflow.log_metric("mse", mse)
            # rmse = np.sqrt(mse)
            # mlflow.log_metric("rmse", rmse)

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
        except Exception as e:
            logger.error(e)
            raise e
