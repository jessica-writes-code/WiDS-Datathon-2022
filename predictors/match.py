from abc import ABC, abstractmethod
import copy
from re import sub
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

pd.options.mode.chained_assignment = None  # default='warn'

class AverageMatchPredictor():
    pass  # TODO

class ModelMatchPredictor():
    """Predictions based on model, either within matched records or
    across dataset (when matched records are unavailable)"""

    MATCH_KEYS = ['floor_area', 'year_built']

    @staticmethod
    def clean_dfs(
        train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Combine data sets
        train_df["_DATASET"] = "train"
        test_df["_DATASET"] = "test"
        df = pd.concat([train_df, test_df])

        # Drop "State_6" from `State_Factor` - It is not represented in the test set
        assert "State_6" not in test_df["State_Factor"]
        df = df[df["State_Factor"] != "State_6"]

        # `floor_area`
        df["log_floor_area"] = np.log(df["floor_area"])

        # `energy_star_rating`
        keys = ["State_Factor", "floor_area", "year_built"]

        avg_df = (
            df[~pd.isna(df["energy_star_rating"])]
            .groupby(keys, as_index=False)
            .mean()[keys + ["energy_star_rating"]]
        )
        avg_df = avg_df.rename(columns={"energy_star_rating": "avg_energy_star_rating"})

        df = df.merge(avg_df, on=keys, how="left")
        df["std_energy_star_rating"] = (
            df["energy_star_rating"]
            .fillna(df["avg_energy_star_rating"])
            .fillna(df[df["_DATASET"] == "train"]["energy_star_rating"].mean())
        )

        # `year_built`
        df.loc[df["year_built"] == 0, "year_built"] = np.NaN
        df["na_year_built"] = pd.isna(df["year_built"])
        df["std_year_built"] = df["year_built"].fillna(
            df[df["_DATASET"] == "train"]["year_built"].mean()
        )
        df["log_year_built"] = np.log(df["std_year_built"])

        # `site_eui`
        df["log_site_eui"] = np.log(df["site_eui"])

        # Down-select to only useful columns
        df = df[
            [
                "id",
                "_DATASET",
                "avg_temp",
                "floor_area",
                "log_floor_area",
                "year_built",
                "std_year_built",
                "log_year_built",
                "na_year_built",
                "std_energy_star_rating",
                "site_eui",
                "log_site_eui",
            ]
        ]

        return (
            df[df["_DATASET"] == "train"].drop("_DATASET", axis=1),
            df[df["_DATASET"] == "test"].drop(
                ["_DATASET", "site_eui", "log_site_eui"], axis=1
            ),
        )

    def fit(self, train_df: pd.DataFrame):

        # 
        self._match_df = train_df.groupby(self.MATCH_KEYS, as_index=False).mean()[
            self.MATCH_KEYS + ['site_eui']
        ]

        # Train a model to be used on test observations that do not
        # match any of our training observations
        # - Note: We are predicting `log_site_eui`
        self._regr = GradientBoostingRegressor()
        self._regr.fit(
            train_df[[
                "avg_temp",
                "floor_area",
                "log_floor_area",
                "std_year_built",
                "log_year_built",
                "na_year_built",
                "std_energy_star_rating",
            ]],
            train_df["site_eui"],
        )

        self.fitted = True

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Comments
        test_df = test_df.merge(self._match_df, on=self.MATCH_KEYS, how="left")

        # TODO: Comments
        predictions = pd.Series(
            self._regr.predict(
                test_df[[
                    "avg_temp",
                    "floor_area",
                    "log_floor_area",
                    "std_year_built",
                    "log_year_built",
                    "na_year_built",
                    "std_energy_star_rating",
                ]],
            )
        )
        test_df["site_eui"] = test_df["site_eui"].fillna(predictions)

        return test_df[['id', 'site_eui']]
