import math

import pandas as pd
from sklearn.model_selection import KFold


def calculate_cv_error(df: pd.DataFrame, model, n_splits: int = 5):
    df = df.sample(frac=1)
    kf = KFold(n_splits=n_splits)
    mse_list = []

    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        model.fit(train_df.drop('site_eui', axis=1), train_df['site_eui'])
        test_df['predicted_site_eui'] = model.predict(test_df.drop('site_eui', axis=1))

        mse_list.append(
            math.sqrt(sum((test_df['predicted_site_eui'] - test_df['site_eui'])**2) / len(test_df))
        )
    
    return mse_list
