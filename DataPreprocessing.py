# DataPreprocessing

from paddle.io import Dataset
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import Compose, Resize, Transpose
import matplotlib.pyplot as plt
import cv2
import numpy as np

class CassavaDataset(Dataset):
    """
    Step 1: Inherit paddle.io.Dataset class
    """

    def __init__(self,
                 train_dir="AMLS_LeafDiseaseClassification/new_train",
                 val_dir="AMLS_LeafDiseaseClassification/new_val",
                 is_test=False,
                 is_train=True):
        """
        Step 2: Implement the constructor, define the data reading method,
        and divide the training and testing data sets
        """

        super(CassavaDataset, self).__init__()

        transform_train = Compose([Resize(size=(600, 800)), Transpose()])
        transform_eval = Compose([Resize(size=(600, 800)), Transpose()])
        train_data_folder = DatasetFolder(train_dir, transform=transform_train)
        val_data_folder = DatasetFolder(val_dir, transform=transform_eval)
        self.is_train = is_train
        self.is_test = is_test
        if self.is_train:
            self.data = train_data_folder
        elif not self.is_train:
            self.data = val_data_folder

    def __getitem__(self, index):
        """
        Step 3: Implement the __getitem__ method, define how to obtain data when specifying an index,
        and return a single piece of data (training data, corresponding label)
        """
        data = np.array(self.data[index][0]).astype('float32')

        if self.is_test:
            return None
        else:
            label = np.array([self.data[index][1]]).astype('int64')
            return data, label

    def __len__(self):
        """
        Step 4: Implement the __len__ method and return the total number of datasets
        """
        return len(self.data)
