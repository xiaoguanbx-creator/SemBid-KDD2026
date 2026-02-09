import os
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')

class NumpyCompatUnpickler(pickle.Unpickler):
    """Map numpy._core.* pickles to numpy.core.* for numpy<2.0."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle_compat(path):
    with open(path, "rb") as file:
        return NumpyCompatUnpickler(file).load()


class TestDataLoader:
    """
    Offline evaluation data loader.
    """

    def __init__(self, file_path="./data/log.csv"):
        """
        Initialize the data loader.
        Args:
            file_path (str): The path to the training data file.

        """
        self.file_path = file_path
        # Cache is bound to the specific file (size + mtime) to avoid stale reuse.
        self.raw_data_path = self._build_cache_path(file_path)
        self.raw_data = self._get_raw_data()
        self.keys, self.test_dict = self._get_test_data_dict()

    def _build_cache_path(self, file_path):
        base_name = os.path.basename(file_path) or "data"
        try:
            stat = os.stat(file_path)
            signature = f"{stat.st_size}_{int(stat.st_mtime)}"
        except OSError:
            signature = "unknown"
        cache_name = f"raw_data_{base_name}_{signature}.pickle"
        return os.path.join(os.path.dirname(file_path), cache_name)

    def _get_raw_data(self):
        """
        Read raw data from a pickle file.

        Returns:
            pd.DataFrame: The raw data as a DataFrame.
        """
        if os.path.exists(self.raw_data_path):
            return load_pickle_compat(self.raw_data_path)
        else:
            tem = pd.read_csv(self.file_path)
            with open(self.raw_data_path, 'wb') as file:
                pickle.dump(tem, file)
            return tem

    def _get_test_data_dict(self):
        """
        Group and sort the raw data by deliveryPeriodIndex and advertiserNumber.

        Returns:
            list: A list of group keys.
            dict: A dictionary with grouped data.

        """
        grouped_data = self.raw_data.sort_values('timeStepIndex').groupby(['deliveryPeriodIndex', 'advertiserNumber'])
        data_dict = {key: group for key, group in grouped_data}
        return list(data_dict.keys()), data_dict

    def mock_data(self, key):
        """
        Get training data based on deliveryPeriodIndex and advertiserNumber, and construct the test data.
        """
        data = self.test_dict[key]
        pValues = data.groupby('timeStepIndex')['pValue'].apply(list).apply(np.array).tolist()
        pValueSigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(list).apply(np.array).tolist()
        leastWinningCosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(list).apply(np.array).tolist()
        num_timeStepIndex = len(pValues)
        budget = data['budget'].iloc[0]
        cpa = data['CPAConstraint'].iloc[0]
        category = data['advertiserCategoryIndex'].iloc[0]
        return num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category


if __name__ == '__main__':
    pass
