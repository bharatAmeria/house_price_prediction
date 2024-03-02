from abc import ABC, abstractmethod
from typing import Union
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import logger


class FeatureEngineeringStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_FE(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class FeatureEngineeringConfig(FeatureEngineeringStrategy):
    def handle_FE(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Feature engineering strategy which preprocesses the data.
        """
        try:
            """ Handling Area With type"""

            # This function extracts the Super Built-up area
            def get_super_built_up_area(text):
                match = re.search(r'Super Built up area (\d+\.?\d*)', text)
                if match:
                    return float(match.group(1))
                return None

            # This function extracts the Built-Up area or Carpet area
            def get_area(text, area_type):
                match = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', text)
                if match:
                    return float(match.group(1))
                return None

            # This function checks if the area is provided in sq.m. and converts it to sqft if needed
            def convert_to_sqft(text, area_value):
                if area_value is None:
                    return None
                match = re.search(r'{} \((\d+\.?\d*) sq.m.\)'.format(area_value), text)
                if match:
                    sq_m_value = float(match.group(1))
                    return sq_m_value * 10.7639  # conversion factor from sq.m. to sqft
                return area_value

            # Extract Super Built-up area and convert to sqft if needed
            data['super_built_up_area'] = data['areaWithType'].apply(get_super_built_up_area)
            data['super_built_up_area'] = data.apply(
                lambda x: convert_to_sqft(x['areaWithType'], x['super_built_up_area']),
                axis=1)

            # Extract Built-Up area and convert to sqft if needed
            data['built_up_area'] = data['areaWithType'].apply(lambda x: get_area(x, 'Built Up area'))
            data['built_up_area'] = data.apply(lambda x: convert_to_sqft(x['areaWithType'], x['built_up_area']), axis=1)

            # Extract Carpet area and convert to sqft if needed
            data['carpet_area'] = data['areaWithType'].apply(lambda x: get_area(x, 'Carpet area'))
            data['carpet_area'] = data.apply(lambda x: convert_to_sqft(x['areaWithType'], x['carpet_area']), axis=1)

            all_nan_df = \
                data[((data['super_built_up_area'].isnull()) & (data['built_up_area'].isnull()) & (
                    data['carpet_area'].isnull()))][
                    ['price', 'property_type', 'area', 'areaWithType', 'super_built_up_area', 'built_up_area',
                     'carpet_area']]

            # Function to extract plot area from 'areaWithType' column
            def extract_plot_area(area_with_type):
                match = re.search(r'Plot area (\d+\.?\d*)', area_with_type)
                return float(match.group(1)) if match else None

            all_nan_df['built_up_area'] = all_nan_df['areaWithType'].apply(extract_plot_area)

            def convert_scale(row):
                if np.isnan(row['area']) or np.isnan(row['built_up_area']):
                    return row['built_up_area']
                else:
                    if round(row['area'] / row['built_up_area']) == 9.0:
                        return row['built_up_area'] * 9
                    elif round(row['area'] / row['built_up_area']) == 11.0:
                        return row['built_up_area'] * 10.7
                    else:
                        return row['built_up_area']

            all_nan_df['built_up_area'] = all_nan_df.apply(convert_scale, axis=1)

            """ Additional Room"""

            new_cols = ['study room', 'servant room', 'store room', 'pooja room', 'others']

            # Populate the new columns based on the "additionalRoom" column
            for col in new_cols:
                data[col] = data['additionalRoom'].str.contains(col).astype(int)

            """ Age of Possession"""

            def categorize_age_possession(value):
                if pd.isna(value):
                    return "Undefined"
                if "0 to 1 Year Old" in value or "Within 6 months" in value or "Within 3 months" in value:
                    return "New Property"
                if "1 to 5 Year Old" in value:
                    return "Relatively New"
                if "5 to 10 Year Old" in value:
                    return "Moderately Old"
                if "10+ Year Old" in value:
                    return "old property"
                if "Under Construction" in value or "By" in value:
                    return "Under Construction"
                try:
                    int(value.split(" ")[-1])
                    return "Under Construction"
                except:
                    return "Undefined"

            data['agePossession'] = data['agePossession'].apply(categorize_age_possession)

            return data
        except Exception as e:
            logger.error(e)
            raise e

class DataDivideStrategy(FeatureEngineeringStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_FE(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(e)
            raise e


class FeatureEngineering:
    """
    Feature engineering class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, strategy: FeatureEngineeringStrategy, data: pd.DataFrame) -> None:
        self.df = data
        self.strategy = strategy

    def handle_FE(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_FE(self.df)
