from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from src import logger


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
            data.insert(loc=3, column='sector',
                        value=data['property_name'].str.split('in').str.get(1).str.replace('Gurgaon', '').str.strip())
            data['sector'] = data['sector'].str.lower()
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

            cleaned_data_v1 = data
            return cleaned_data_v1

        except Exception as e:
            logger.error(e)
            raise e

class DataCleaning(DataStrategy):
    """
    Data cleaning class which preprocesses the data
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
