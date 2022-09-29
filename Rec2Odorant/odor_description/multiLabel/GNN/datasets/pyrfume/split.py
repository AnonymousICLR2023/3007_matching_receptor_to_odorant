import os
import pandas
from shutil import copy

from Rec2Odorant.odor_description.base_cross_validation import BaseCVSplit

class Random_split(BaseCVSplit):
    """
    """
    def func_split_data(self, data, seed, **kwargs):
        """
        function that takes data as input and outputs test_data, validation_data
        and train_data dataframes.

        Needs to be overwriten by user. By default calls self.random_data.

        Paramters:
        ----------
        data : pandas.DataFrame
            dataframe returned by self.func_data 
        """
        return self.random_data(data, seed, **kwargs)


if __name__ == '__main__':
    import time
    split = Random_split(data_dir = os.path.join('JAX','multiLabel', 'GNN', 'Data', 'Test', 'pyrfume_base_20220908-104134'), 
                            seed = int(time.time()), 
                            split_kwargs = {'valid_ratio' : 0.1,
                                            'test_ratio' : 0.0})
    split.CV_split()