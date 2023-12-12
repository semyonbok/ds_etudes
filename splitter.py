import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
# from sklearn.utils.estimator_checks import check_estimator


class Splitter(BaseEstimator, TransformerMixin):
    def __init__(self, sep="_"):
        self.sep = sep

    def fit(self, X, y=None):
        assert X.shape[1] == 1, \
            "\nX must be a 2-D array with one column."
        *_, nonnan_entry_length = self._split(X)
        self.n_features_out_ = nonnan_entry_length.unique()[0]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.values[0]
        else:
            self.feature_names_in_ = ""
        return self

    def transform(self, X, cast_to_float=True):
        check_is_fitted(self)
        nan_mask, X_split, nonnan_entry_length = self._split(X)
        assert nonnan_entry_length.unique()[0] == self.n_features_out_, \
            f"\nSplit with '{self.sep}' yields different n of features than seen in `fit`."

        out = np.tile(np.array(np.nan, dtype=object),
                      (len(X), self.n_features_out_))
        out[~nan_mask] = np.vstack(X_split[~nan_mask])

        if cast_to_float:
            for i in range(self.n_features_out_):
                try:
                    out[:, i] = out[:, i].astype(float)
                except ValueError:
                    continue
        return out

    def get_feature_names_out(self, names=None):
        check_is_fitted(self)
        if hasattr(self, "feature_names_in_"):
            return [f"{self.feature_names_in_}_S{i}" for i in range(self.n_features_out_)]
        else:
            return [f"S{i}" for i in range(self.n_features_out_)]

    def _split(self, X):
        X = check_array(X, dtype=str, force_all_finite="allow-nan")
        nan_mask = (X.ravel() == "nan")
        X_split = pd.Series(X.ravel()).str.split(self.sep).values
        nonnan_entry_length = pd.Series(X_split[~nan_mask]).apply(len)

        assert nonnan_entry_length.nunique() == 1, \
            f"\nSplit with '{self.sep}' yields varying n of features per entry."

        return nan_mask, X_split, nonnan_entry_length
