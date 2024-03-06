from src import logger
from src.entity.config_entity import ModelTrainerConfig
from src.components.model_devlopment import LinearRegression, XGBoostModel, RandomForestModel, RandomForestRegressor
from src.components.evaluation import R2Score
from src.utils.common import save_object, evaluate_models


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model_dir = self.config.model_dir

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBoostModel(),
                "RandomForestModel": RandomForestModel()
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=self.config.param)

            """ To get best model score from dict """
            best_model_score = max(sorted(model_report.values()))

            """ To get best model name from dict """
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise logger.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.config.model_dir,
                obj=best_model
            )

            r2_square = R2Score()
            return r2_square
        except Exception as e:
            logger.error(e)
            raise e
