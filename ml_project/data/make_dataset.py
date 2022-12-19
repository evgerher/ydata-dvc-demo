from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities import SplittingParams
from ml_project.features.make_features import Dataset


def read_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(data: Dataset, params: SplittingParams) -> Tuple[Dataset, Dataset]:
    train_data_X, val_data_X, train_data_Y, val_data_Y = train_test_split(
        data.X, data.Y, test_size=params.val_size, random_state=params.random_state
    )
    train_ds = Dataset(train_data_X, train_data_Y)
    val_ds = Dataset(val_data_X, val_data_Y)
    return train_ds, val_ds
