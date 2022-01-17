from abc import ABC, abstractmethod
import copy
from typing import List

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

TARGET_COL_NAME = '__TARGET'
ID_COL_NAME = '__ID'


class AverageMatchPredictor():

    def __init__(self, keys=['year_built', 'floor_area']):
        self.keys = keys
        self.fitted = False

    def _fit_matches(self, X: pd.DataFrame, y: pd.Series):
        """"""
        self._match_df = X.groupby(self.keys, as_index=False).mean()[self.keys + [TARGET_COL_NAME]]

    def _fit_nonmatches(self, X: pd.DataFrame, y: pd.Series):
        """"""
        self._avg_value = sum(y) / len(y)

    def _is_match(self, X: pd.DataFrame) -> List[bool]:
        """"""
        tmp = X.merge(self._match_df, on=self.keys, how='left')
        return [not item for i, item in pd.isna(tmp[TARGET_COL_NAME]).iteritems()]

    def _predict_matches(self, X: pd.DataFrame) -> pd.Series:
        """"""
        return X.merge(self._match_df, on=self.keys, how='inner')[[TARGET_COL_NAME, ID_COL_NAME]]

    def _predict_nonmatches(self, X: pd.DataFrame) -> pd.Series:
        """"""
        temp = X[[ID_COL_NAME]]
        temp[TARGET_COL_NAME] = self._avg_value
        return temp

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """TODO"""
        assert TARGET_COL_NAME not in X.columns
        X = copy.deepcopy(X)
        X[TARGET_COL_NAME] = y
        self._fit_matches(X, y)
        self._fit_nonmatches(X, y)
        self.fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """TODO"""
        if not self.fitted:
            raise ValueError
        
        X = copy.deepcopy(X)
        X[ID_COL_NAME] = X.index
        is_match = self._is_match(X)
        is_not_match = [not x for x in is_match]

        matched_df = self._predict_matches(X[is_match])
        nonmatched_df = self._predict_nonmatches(X[is_not_match])
        full_df = pd.concat([matched_df, nonmatched_df], axis=0).sort_values(ID_COL_NAME)

        assert len(full_df) == len(X)

        return list(full_df[TARGET_COL_NAME])
