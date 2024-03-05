import re
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import logger


class FeatureEngineeringStrategy:
    """
    Abstract Class defining strategy for handling data
    """
    def handle_FE(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class FeatureEngineeringConfig(FeatureEngineeringStrategy):
    def handle_FE(self, data: pd.DataFrame) -> pd.DataFrame:
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

            """ Furnishing Details"""
            # Extract all unique furnishings from the furnishDetails column
            all_furnishings = []
            for detail in data['furnishDetails'].dropna():
                furnishings = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
                all_furnishings.extend(furnishings)
            unique_furnishings = list(set(all_furnishings))

            # Define a function to extract the count of a furnishing from the furnishDetails
            def get_furnishing_count(details, furnishing):
                if isinstance(details, str):
                    if f"No {furnishing}" in details:
                        return 0
                    pattern = re.compile(f"(\d+) {furnishing}")
                    match = pattern.search(details)
                    if match:
                        return int(match.group(1))
                    elif furnishing in details:
                        return 1
                return 0

            # Simplify the furnishings list by removing "No" prefix and numbers
            columns_to_include = [re.sub(r'No |\d+', '', furnishing).strip() for furnishing in unique_furnishings]
            columns_to_include = list(set(columns_to_include))  # Get unique furnishings
            columns_to_include = [furnishing for furnishing in columns_to_include if furnishing]  # Remove empty strings

            # Create new columns for each unique furnishing and populate with counts
            for furnishing in columns_to_include:
                data[furnishing] = data['furnishDetails'].apply(lambda x: get_furnishing_count(x, furnishing))

            # Create the new dataframe with the required columns
            furnishings_df = data[['furnishDetails'] + columns_to_include]
            furnishings_df.drop(columns=['furnishDetails'], inplace=True)

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(furnishings_df)

            wcss_reduced = []

            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(scaled_data)
                wcss_reduced.append(kmeans.inertia_)

            n_clusters = 3

            # Fit the KMeans model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(scaled_data)

            # Predict the cluster assignments for each row
            cluster_assignments = kmeans.predict(scaled_data)

            data = data.iloc[:, :-18]
            data['furnishing_type'] = cluster_assignments

            return data
        except Exception as e:
            logger.error(e)
            raise e

class OutlierTreatment(FeatureEngineeringStrategy):
    def handle_FE(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the IQR for the 'price' column
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

        # Displaying the number of outliers and some statistics
        outliers_price_stats = outliers['price'].describe()

        # Calculate the IQR for the 'price' column
        Q1 = df['price_per_sqft'].quantile(0.25)
        Q3 = df['price_per_sqft'].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

        # Displaying the number of outliers and some statistics

        outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x: x * 9 if x < 1000 else x)
        outliers_sqft['price_per_sqft'] = round((outliers_sqft['price'] * 10000000) / outliers_sqft['area'])
        data.update(outliers_sqft)

        """Area"""
        df.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True)
        df.loc[48, 'area'] = 115 * 9
        df.loc[300, 'area'] = 7250
        df.loc[2666, 'area'] = 5800
        df.loc[1358, 'area'] = 2660
        df.loc[3195, 'area'] = 2850
        df.loc[2131, 'area'] = 1812
        df.loc[3088, 'area'] = 2160
        df.loc[3444, 'area'] = 1175

        df.loc[2131, 'carpet_area'] = 1812
        df['price_per_sqft'] = round((df['price'] * 10000000) / df['area'])
        x = df[df['price_per_sqft'] <= 20000]
        (x['area'] / x['bedRoom']).quantile(0.02)

        sbc_df = df[
            ~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

        sbc_df['built_up_area'].fillna(
            round(((sbc_df['super_built_up_area'] / 1.105) + (sbc_df['carpet_area'] / 0.9)) / 2),
            inplace=True)
        df.update(sbc_df)

        # sb present c is null built up null
        sb_df = df[
            ~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]

        sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area'] / 1.105), inplace=True)
        df.update(sb_df)

        # sb null c is present built up null
        c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
        c_df['built_up_area'].fillna(round(c_df['carpet_area'] / 0.9), inplace=True)
        df.update(c_df)

        anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price', 'area', 'built_up_area']]
        anamoly_df['built_up_area'] = anamoly_df['area']
        df.update(anamoly_df)

        df['floorNum'].fillna(2.0, inplace=True)
        df.drop(columns=['facing'], inplace=True)
        df.drop(index=[2536], inplace=True)

        def mode_based_imputation(row):
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])][
                    'agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0]
                else:
                    return np.nan
            else:
                return row['agePossession']

        df['agePossession'] = df.apply(mode_based_imputation, axis=1)

        def mode_based_imputation2(row):
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0]
                else:
                    return np.nan
            else:
                return row['agePossession']

        df['agePossession'] = df.apply(mode_based_imputation2, axis=1)

        def mode_based_imputation3(row):
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0]
                else:
                    return np.nan
            else:
                return row['agePossession']

        df['agePossession'] = df.apply(mode_based_imputation3, axis=1)

        return df

class Appartments(FeatureEngineeringStrategy):
    pass

class FeatureSelection(FeatureEngineeringStrategy):
    def handle_FE(self, df: pd.DataFrame) -> pd.DataFrame:
        train_df = df.drop(columns=['society', 'price_per_sqft'])
        train_df.corr()['price'].sort_values(ascending=False)

        return df


class DataDivideStrategy(FeatureEngineeringStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_FE(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("price", axis=1)
            y = data["price"]
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

    def __init__(self, data: pd.DataFrame, strategy: FeatureEngineeringStrategy) -> None:
        self.cleaned_data = data
        self.strategy = strategy

    def handle_FE(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_FE(self.cleaned_data)



