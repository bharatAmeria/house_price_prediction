import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src import logger
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.directory = self.config.directory

    def get_data_transformer_object(self):

        """ This function is responsible for data transformation """
        try:

            """ Ordinal Encoding"""
            numerical_columns = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']

            categorical_columns = ['property_type', 'sector', 'balcony', 'agePossession', 'furnishing_type',
                                   'luxury_category', 'floor_category']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())

                ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("Ordinal Encoder", OrdinalEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logger.info(f"Categorical columns: {categorical_columns}"),
            logger.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ])

            return preprocessor

        except Exception as e:
            logger.exception(e)
            raise e

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price"
            numerical_columns = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_([input_feature_train_arr, np.array(target_feature_train_df)])
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info(f"Saved preprocessing object.")

            save_object(file_path=self.config.directory, obj=preprocessing_obj)

            return (train_arr,
                    test_arr,
                    self.config.directory
                    )

        except Exception as e:
            logger.exception(e)
            raise e









