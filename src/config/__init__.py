import pandas as pd


def data_dir() -> pd.DataFrame:
    data = pd.read_csv("artifacts/data_ingestion/gurgaon_properties/gurgaon_properties.csv")
    return data
