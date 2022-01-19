from typing import Tuple

import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    return train_df, test_df


def run_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_df_clean, test_df_clean = model.clean_dfs(train_df, test_df)
    model.fit(train_df_clean)
    return model.predict(test_df_clean)
