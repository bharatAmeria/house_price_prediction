import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src import logger
from src.components.model_devlopment import RandomForestModel, XGBoostModel, LinearRegressionModel, HyperparameterTuner
from src.constants.constants import ModelNameConfig


class Model:
    def __init__(self):
        pass

    @staticmethod
    def main(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
             config: ModelNameConfig) -> RegressorMixin:
        try:
            model = None
            tuner = None

            if config.model_name == "randomForest":
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

            tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)

            if config.fine_tuning:
                best_params = tuner.optimize()
                trained_model = model.train(X_train, y_train, **best_params)
            else:
                trained_model = model.train(X_train, y_train)
            return trained_model
        except Exception as e:
            logger.error(e)
            raise e