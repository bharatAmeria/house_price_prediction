import pandas as pd
from src.components.data_cleaning import DataCleaning, DataPreprocessStrategy
from src import logger

def get_data_for_test():
    try:
        df = pd.read_csv("artifacts/data_ingestion/gurgaon_properties/gurgaon_properties.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["price"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logger.error(e)
        raise e