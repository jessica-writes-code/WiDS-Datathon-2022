from typing import Tuple

import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    return train_df, test_df
