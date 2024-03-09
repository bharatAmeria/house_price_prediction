import ast
import re
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from src import logger
from src.config.configuration import DataDividerConfig


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

            data[['price', 'property_type', 'area', 'areaWithType', 'super_built_up_area', 'built_up_area',
                'carpet_area']].sample(5)

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

            app_df = pd.read_csv('artifacts/data_ingestion/gurgaon_properties/appartments.csv')

            app_df['PropertyName'] = app_df['PropertyName'].str.lower()
            temp_df = data[data['features'].isnull()]

            x = temp_df.merge(app_df, left_on='society', right_on='PropertyName', how='left')['TopFacilities']
            data.loc[temp_df.index, 'features'] = x.values

            # Convert the string representation of lists in the 'features' column to actual lists
            data['features_list'] = data['features'].apply(
                lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])

            # Use MultiLabelBinarizer to convert the features list into a binary matrix
            mlb = MultiLabelBinarizer()
            features_binary_matrix = mlb.fit_transform(data['features_list'])

            # Convert the binary matrix into a DataFrame
            features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)

            wcss_reduced = []

            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(features_binary_df)
                wcss_reduced.append(kmeans.inertia_)

            weights = {'24/7 Power Backup': 8, '24/7 Water Supply': 4, '24x7 Security': 7, 'ATM': 4,
                       'Aerobics Centre': 6,
                       'Airy Rooms': 8, 'Amphitheatre': 7, 'Badminton Court': 7, 'Banquet Hall': 8,
                       'Bar/Chill-Out Lounge': 9,
                       'Barbecue': 7, 'Basketball Court': 7, 'Billiards': 7, 'Bowling Alley': 8, 'Business Lounge': 9,
                       'CCTV Camera Security': 8, 'Cafeteria': 6, 'Car Parking': 6, 'Card Room': 6,
                       'Centrally Air Conditioned': 9, 'Changing Area': 6, "Children's Play Area": 7, 'Cigar Lounge': 9,
                       'Clinic': 5, 'Club House': 9, 'Concierge Service': 9, 'Conference room': 8, 'Creche/Day care': 7,
                       'Cricket Pitch': 7, 'Doctor on Call': 6, 'Earthquake Resistant': 5, 'Entrance Lobby': 7,
                       'False Ceiling Lighting': 6, 'Feng Shui / Vaastu Compliant': 5, 'Fire Fighting Systems': 8,
                       'Fitness Centre / GYM': 8, 'Flower Garden': 7, 'Food Court': 6, 'Foosball': 5, 'Football': 7,
                       'Fountain': 7, 'Gated Community': 7, 'Golf Course': 10, 'Grocery Shop': 6, 'Gymnasium': 8,
                       'High Ceiling Height': 8, 'High Speed Elevators': 8, 'Infinity Pool': 9, 'Intercom Facility': 7,
                       'Internal Street Lights': 6, 'Internet/wi-fi connectivity': 7, 'Jacuzzi': 9, 'Jogging Track': 7,
                       'Landscape Garden': 8, 'Laundry': 6, 'Lawn Tennis Court': 8, 'Library': 8, 'Lounge': 8,
                       'Low Density Society': 7, 'Maintenance Staff': 6, 'Manicured Garden': 7, 'Medical Centre': 5,
                       'Milk Booth': 4, 'Mini Theatre': 9, 'Multipurpose Court': 7, 'Multipurpose Hall': 7,
                       'Natural Light': 8, 'Natural Pond': 7, 'Park': 8, 'Party Lawn': 8,
                       'Piped Gas': 7, 'Pool Table': 7, 'Power Back up Lift': 8, 'Private Garden / Terrace': 9,
                       'Property Staff': 7, 'RO System': 7, 'Rain Water Harvesting': 7, 'Reading Lounge': 8,
                       'Restaurant': 8,
                       'Salon': 8, 'Sauna': 9, 'Security / Fire Alarm': 9, 'Security Personnel': 9,
                       'Separate entry for servant room': 8, 'Sewage Treatment Plant': 6, 'Shopping Centre': 7,
                       'Skating Rink': 7, 'Solar Lighting': 6, 'Solar Water Heating': 7, 'Spa': 9,
                       'Spacious Interiors': 9,
                       'Squash Court': 8, 'Steam Room': 9, 'Sun Deck': 8,
                       'Swimming Pool': 8, 'Temple': 5, 'Theatre': 9, 'Toddler Pool': 7, 'Valet Parking': 9,
                       'Video Door Security': 9, 'Visitor Parking': 7, 'Water Softener Plant': 7, 'Water Storage': 7,
                       'Water purifier': 7, 'Yoga/Meditation Area': 7
                       }
            # Calculate luxury score for each row
            luxury_score = features_binary_df[list(weights.keys())].multiply(list(weights.values())).sum(axis=1)

            data['luxury_score'] = luxury_score
            # cols to drop -> nearbyLocations,furnishDetails, features,features_list, additionalRoom
            data.drop(columns=['nearbyLocations', 'furnishDetails', 'features', 'features_list', 'additionalRoom'],
                      inplace=True)

            feature_engineered_data = data

            return feature_engineered_data
        except Exception as e:
            logger.error(e)
            raise e

class OutlierTreatment(FeatureEngineeringStrategy):
    def handle_FE(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate the IQR for the 'price' column
        Q1 = data['price'].quantile(0.25)
        Q3 = data['price'].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = data[(data['price'] < lower_bound) | (data['price'] > upper_bound)]

        # Displaying the number of outliers and some statistics
        num_outliers = outliers.shape[0]
        outliers_price_stats = outliers['price'].describe()

        outliers.sort_values('price', ascending=False)

        # Calculate the IQR for the 'price' column
        Q1 = data['price_per_sqft'].quantile(0.25)
        Q3 = data['price_per_sqft'].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers_sqft = data[(data['price_per_sqft'] < lower_bound) | (data['price_per_sqft'] > upper_bound)]

        # Displaying the number of outliers and some statistics

        outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x: x * 9 if x < 1000 else x)
        outliers_sqft['price_per_sqft'] = round((outliers_sqft['price'] * 10000000) / outliers_sqft['area'])
        data.update(outliers_sqft)

        """Area"""
        data = data[data['area'] < 100000]
        data[data['area'] > 10000].sort_values('area', ascending=False)

        data.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True)
        data[data['area'] > 10000].sort_values('area', ascending=False)

        data.loc[48, 'area'] = 115 * 9
        data.loc[300, 'area'] = 7250
        data.loc[2666, 'area'] = 5800
        data.loc[1358, 'area'] = 2660
        data.loc[3195, 'area'] = 2850
        data.loc[2131, 'area'] = 1812
        data.loc[3088, 'area'] = 2160
        data.loc[3444, 'area'] = 1175

        """Bedroom"""
        data[data['bedRoom'] > 10].sort_values('bedRoom', ascending=False)
        data = data[data['bedRoom'] <= 10]

        """Bathroom"""
        data[data['bathroom'] > 10].sort_values('bathroom', ascending=False)

        """Super Built up area"""
        data.loc[2131, 'carpet_area'] = 1812
        data['price_per_sqft'] = round((data['price'] * 10000000) / data['area'])
        x = data[data['price_per_sqft'] <= 20000]
        (x['area'] / x['bedRoom']).quantile(0.02)

        outlier_treated_data = data

        return outlier_treated_data


class MissingValueImputation(FeatureEngineeringStrategy):

    def handle_FE(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            print(data.columns)
            all_present_df = data[
                ~((data['super_built_up_area'].isnull()) | (data['built_up_area'].isnull()) | (
                    data['carpet_area'].isnull()))]

            super_to_built_up_ratio = (all_present_df['super_built_up_area'] / all_present_df['built_up_area']).median()

            carpet_to_built_up_ratio = (all_present_df['carpet_area'] / all_present_df['built_up_area']).median()
            print(super_to_built_up_ratio, carpet_to_built_up_ratio)

            # both present built up null
            sbc_df = data[~(data['super_built_up_area'].isnull()) & (data['built_up_area'].isnull()) & ~(
                data['carpet_area'].isnull())]
            sbc_df.head()

            sbc_df['built_up_area'].fillna(
                round(((sbc_df['super_built_up_area'] / 1.105) + (sbc_df['carpet_area'] / 0.9)) / 2), inplace=True)
            data.update(sbc_df)

            # sb present c is null built up null
            sb_df = data[~(data['super_built_up_area'].isnull()) & (data['built_up_area'].isnull()) & (
                data['carpet_area'].isnull())]
            sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area'] / 1.105), inplace=True)
            data.update(sb_df)

            # sb null c is present built up null
            c_df = data[(data['super_built_up_area'].isnull()) & (data['built_up_area'].isnull()) & ~(
                data['carpet_area'].isnull())]
            c_df['built_up_area'].fillna(round(c_df['carpet_area'] / 0.9), inplace=True)
            data.update(c_df)

            anamoly_df = data[(data['built_up_area'] < 2000) & (data['price'] > 2.5)][
                ['price', 'area', 'built_up_area']]
            anamoly_df['built_up_area'] = anamoly_df['area']
            data.update(anamoly_df)

            # 'area_room_ratio' was removed from below line need ot be adjusted
            data.drop(columns=['area', 'areaWithType', 'super_built_up_area', 'carpet_area' ],
                      inplace=True)

            """Floor number"""
            data[data['property_type'] == 'house']['floorNum'].median()
            data['floorNum'].fillna(2.0, inplace=True)

            """Facing"""
            data.drop(columns=['facing'], inplace=True)
            data.drop(index=[2536], inplace=True)

            """Age of Possession"""

            def mode_based_imputation(row):
                if row['agePossession'] == 'Undefined':
                    mode_value = \
                        data[(data['sector'] == row['sector']) & (data['property_type'] == row['property_type'])][
                            'agePossession'].mode()
                    # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                    if not mode_value.empty:
                        return mode_value.iloc[0]
                    else:
                        return np.nan
                else:
                    return row['agePossession']

            data['agePossession'] = data.apply(mode_based_imputation, axis=1)

            def mode_based_imputation2(row):
                if row['agePossession'] == 'Undefined':
                    mode_value = data[(data['sector'] == row['sector'])]['agePossession'].mode()
                    # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                    if not mode_value.empty:
                        return mode_value.iloc[0]
                    else:
                        return np.nan
                else:
                    return row['agePossession']

            data['agePossession'] = data.apply(mode_based_imputation2, axis=1)

            def mode_based_imputation3(row):
                if row['agePossession'] == 'Undefined':
                    mode_value = data[(data['property_type'] == row['property_type'])]['agePossession'].mode()
                    # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                    if not mode_value.empty:
                        return mode_value.iloc[0]
                    else:
                        return np.nan
                else:
                    return row['agePossession']

            data['agePossession'] = data.apply(mode_based_imputation3, axis=1)

            missing_value_imputed_data = data

            return missing_value_imputed_data

        except Exception as e:
            logger.error(e)
            raise e


class FeatureSelection(FeatureEngineeringStrategy):
    def handle_FE(self, data: pd.DataFrame) -> pd.DataFrame:
        train_df = data.drop(columns=['society', 'price_per_sqft'])

        train_df.corr()['price'].sort_values(ascending=False)

        feature_Selection_data = data
        return feature_Selection_data


class DataDivideStrategy(FeatureEngineeringStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def __init__(self):
        """Initialize the data ingestion class."""
        pass

    def handle_FE(self, data: pd.DataFrame):
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("price", axis=1)
            y = data["price"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train.to_csv("artifacts/data_ingestion/train.csv", index=False, header=True)

            X_test.to_csv("artifacts/data_ingestion/test.csv", index=False, header=True)

            logger.info("Ingestion of the data is completed")

            return X_train, y_train, X_test, y_test

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
