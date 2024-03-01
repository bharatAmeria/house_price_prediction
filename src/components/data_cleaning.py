import re

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from src import logger

import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values,
        and converts the data type to float.
        """
        try:
            data = data.drop(columns=['link', 'property_id'], inplace=True)

            data.rename(columns={'area': 'price_per_sqft'}, inplace=True)

            data['society'] = data['society'].apply(
                lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()

            def treat_price(x):
                if type(x) == float:
                    return x
                else:
                    if x[1] == 'Lac':
                        return round(float(x[0]) / 100, 2)
                    else:
                        return round(float(x[0]), 2)

            data['price'] = data['price'].str.split(' ').apply(treat_price)
            data['price_per_sqft'] = data['price_per_sqft'].str.split('/').str.get(0). \
                str.replace('₹', '').str.replace(',', '').str.strip().astype('float')

            data['bedRoom'] = data['bedRoom'].str.split(' ').str.get(0).astype('int')
            data['bathroom'] = data['bathroom'].str.split(' ').str.get(0).astype('int')
            data['balcony'] = data['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

            data['additionalRoom'].fillna('not available', inplace=True)

            data['sector'] = data['sector'].str.lower()
            data['additionalRoom'] = data['additionalRoom'].str.lower()
            data['floorNum'] = data['floorNum'].str.split(' ').str.get(0). \
                replace('Ground', '0').str.replace('Basement', '-1').str.replace('Lower', '0').str.extract(r'(\d+)')
            data['facing'].fillna('NA', inplace=True)

            data.insert(loc=4, column='area', value=round((data['price'] * 10000000) / data['price_per_sqft']))
            data.insert(loc=1, column='property_type', value='flat')
            data.insert(loc=3, column='sector',
                        value=data['property_name'].str.split('in').str.get(1).str.replace('Gurgaon', '').str.strip())

            data['society'] = data['society'].str.replace('nan', 'independent')

            data['sector'] = data['sector'].str.replace('dharam colony', 'sector 12')
            data['sector'] = data['sector'].str.replace('krishna colony', 'sector 7')
            data['sector'] = data['sector'].str.replace('suncity', 'sector 54')
            data['sector'] = data['sector'].str.replace('prem nagar', 'sector 13')
            data['sector'] = data['sector'].str.replace('mg road', 'sector 28')
            data['sector'] = data['sector'].str.replace('gandhi nagar', 'sector 28')
            data['sector'] = data['sector'].str.replace('laxmi garden', 'sector 11')
            data['sector'] = data['sector'].str.replace('shakti nagar', 'sector 11')

            data['sector'] = data['sector'].str.replace('baldev nagar', 'sector 7')
            data['sector'] = data['sector'].str.replace('shivpuri', 'sector 7')
            data['sector'] = data['sector'].str.replace('garhi harsaru', 'sector 17')
            data['sector'] = data['sector'].str.replace('imt manesar', 'manesar')
            data['sector'] = data['sector'].str.replace('adarsh nagar', 'sector 12')
            data['sector'] = data['sector'].str.replace('shivaji nagar', 'sector 11')
            data['sector'] = data['sector'].str.replace('bhim nagar', 'sector 6')
            data['sector'] = data['sector'].str.replace('madanpuri', 'sector 7')

            data['sector'] = data['sector'].str.replace('saraswati vihar', 'sector 28')
            data['sector'] = data['sector'].str.replace('arjun nagar', 'sector 8')
            data['sector'] = data['sector'].str.replace('ravi nagar', 'sector 9')
            data['sector'] = data['sector'].str.replace('vishnu garden', 'sector 105')
            data['sector'] = data['sector'].str.replace('bhondsi', 'sector 11')
            data['sector'] = data['sector'].str.replace('surya vihar', 'sector 21')
            data['sector'] = data['sector'].str.replace('devilal colony', 'sector 9')
            data['sector'] = data['sector'].str.replace('valley view estate', 'gwal pahari')

            data['sector'] = data['sector'].str.replace('mehrauli  road', 'sector 14')
            data['sector'] = data['sector'].str.replace('jyoti park', 'sector 7')
            data['sector'] = data['sector'].str.replace('ansal plaza', 'sector 23')
            data['sector'] = data['sector'].str.replace('dayanand colony', 'sector 6')
            data['sector'] = data['sector'].str.replace('sushant lok phase 2', 'sector 55')
            data['sector'] = data['sector'].str.replace('chakkarpur', 'sector 28')
            data['sector'] = data['sector'].str.replace('greenwood city', 'sector 45')
            data['sector'] = data['sector'].str.replace('subhash nagar', 'sector 12')

            data['sector'] = data['sector'].str.replace('sohna road road', 'sohna road')
            data['sector'] = data['sector'].str.replace('malibu town', 'sector 47')
            data['sector'] = data['sector'].str.replace('surat nagar 1', 'sector 104')
            data['sector'] = data['sector'].str.replace('new colony', 'sector 7')
            data['sector'] = data['sector'].str.replace('mianwali colony', 'sector 12')
            data['sector'] = data['sector'].str.replace('jacobpura', 'sector 12')
            data['sector'] = data['sector'].str.replace('rajiv nagar', 'sector 13')
            data['sector'] = data['sector'].str.replace('ashok vihar', 'sector 3')

            data['sector'] = data['sector'].str.replace('dlf phase 1', 'sector 26')
            data['sector'] = data['sector'].str.replace('nirvana country', 'sector 50')
            data['sector'] = data['sector'].str.replace('palam vihar', 'sector 2')
            data['sector'] = data['sector'].str.replace('dlf phase 2', 'sector 25')
            data['sector'] = data['sector'].str.replace('sushant lok phase 1', 'sector 43')
            data['sector'] = data['sector'].str.replace('laxman vihar', 'sector 4')
            data['sector'] = data['sector'].str.replace('dlf phase 4', 'sector 28')
            data['sector'] = data['sector'].str.replace('dlf phase 3', 'sector 24')

            data['sector'] = data['sector'].str.replace('sushant lok phase 3', 'sector 57')
            data['sector'] = data['sector'].str.replace('dlf phase 5', 'sector 43')
            data['sector'] = data['sector'].str.replace('rajendra park', 'sector 105')
            data['sector'] = data['sector'].str.replace('uppals southend', 'sector 49')
            data['sector'] = data['sector'].str.replace('sohna', 'sohna road')
            data['sector'] = data['sector'].str.replace('ashok vihar phase 3 extension', 'sector 5')
            data['sector'] = data['sector'].str.replace('south city 1', 'sector 41')
            data['sector'] = data['sector'].str.replace('ashok vihar phase 2', 'sector 5')

            a = data['sector'].value_counts()[data['sector'].value_counts() >= 3]
            data = data[data['sector'].isin(a.index)]

            data['sector'] = data['sector'].str.replace('sector 95a', 'sector 95')
            data['sector'] = data['sector'].str.replace('sector 23a', 'sector 23')
            data['sector'] = data['sector'].str.replace('sector 12a', 'sector 12')
            data['sector'] = data['sector'].str.replace('sector 3a', 'sector 3')
            data['sector'] = data['sector'].str.replace('sector 110 a', 'sector 110')
            data['sector'] = data['sector'].str.replace('patel nagar', 'sector 15')
            data['sector'] = data['sector'].str.replace('a block sector 43', 'sector 43')
            data['sector'] = data['sector'].str.replace('maruti kunj', 'sector 12')
            data['sector'] = data['sector'].str.replace('b block sector 43', 'sector 43')

            data['sector'] = data['sector'].str.replace('sector-33 sohna road', 'sector 33')
            data['sector'] = data['sector'].str.replace('sector 1 manesar', 'manesar')
            data['sector'] = data['sector'].str.replace('sector 4 phase 2', 'sector 4')
            data['sector'] = data['sector'].str.replace('sector 1a manesar', 'manesar')
            data['sector'] = data['sector'].str.replace('c block sector 43', 'sector 43')
            data['sector'] = data['sector'].str.replace('sector 89 a', 'sector 89')
            data['sector'] = data['sector'].str.replace('sector 2 extension', 'sector 2')
            data['sector'] = data['sector'].str.replace('sector 36 sohna road', 'sector 36')

            data.loc[955, 'sector'] = 'sector 37'
            data.loc[2800, 'sector'] = 'sector 92'
            data.loc[2838, 'sector'] = 'sector 90'
            data.loc[2857, 'sector'] = 'sector 76'

            data.loc[[311, 1072, 1486, 3040, 3875], 'sector'] = 'sector 110'

            data.drop(columns=['property_name', 'address', 'description', 'rating'], inplace=True)

            return data
        except Exception as e:
            logger.error(e)
            raise e


class FeatureEngineering(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
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
                    # For entries like 'May 2024'
                    int(value.split(" ")[-1])
                    return "Under Construction"
                except:
                    return "Undefined"

            data['agePossession'] = data['agePossession'].apply(categorize_age_possession)

            # Extract all unique furnishings from the furnishDetails column
            all_furnishings = []
            for detail in df['furnishDetails'].dropna():
                furnishings = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
                all_furnishings.extend(furnishings)
                unique_furnishings = list(set(all_furnishings))

            # Define a function to extract the count of a furnishing from the furnishDetails
            def get_furnishing_count(details, furnishing):
                
            return data
        except Exception as e:
            logger.error(e)
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
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


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
