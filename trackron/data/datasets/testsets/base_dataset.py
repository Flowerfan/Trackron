import numpy as np
from torch.utils.data import Dataset
# from evaluation.evaluation.environment import env_settings
from collections import OrderedDict


class BaseDataset(Dataset):
    """Base class for all datasets."""
    def __init__(self):
        pass
        # self.config = cfg 

    def __len__(self):
        """Overload this function in your dataset. This should return number of sequences in the dataset."""
        raise NotImplementedError

    def get_sequence_list(self):
        """Overload this in your dataset. Should return the list of sequences in the dataset."""
        raise NotImplementedError
    
    def __getitem__(self, index):
        """[summary]

        Args:
            index ([type]): [description]

        Raises:
            IndexError: [description]

        Returns:
            [type]: [description]
        """
        raise None

