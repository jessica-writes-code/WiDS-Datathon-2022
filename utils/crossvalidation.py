import math

import pandas as pd
from sklearn.model_selection import KFold

from utils import helpers


def calculate_cv_error(df: pd.DataFrame, model, n_splits: int = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    mse_list = []

    for train_index, test_index in kf.split(df):
        # Train
        train_df = df.iloc[train_index]

        # Test
        test_df = df.iloc[test_index]
        test_df = test_df[
            test_df["State_Factor"] != "State_6"
        ]  # test set has no `State_6` entries
        actuals = test_df.reset_index()["site_eui"]

        # Model -> Predictions
        predictions_df = helpers.run_model(model, train_df, test_df)

        mse_list.append(
            math.sqrt(sum((predictions_df["site_eui"] - actuals) ** 2) / len(test_df))
        )

    return mse_list
