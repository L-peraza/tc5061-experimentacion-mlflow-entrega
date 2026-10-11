import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BankCleaner(TransformerMixin, BaseEstimator):
    def __init__(self, unknown_value="unknown"):
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.unknown_columns_ = [
            name for name in ("job", "education", "contact") if name in X
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if not X.columns.equals(pd.Index(self.feature_names_in_)):
            raise ValueError("Las columnas deben coincidir con entrenamiento.")
        result = X.copy()
        columns = self.unknown_columns_
        result[columns] = result[columns].replace(self.unknown_value, np.nan)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()
