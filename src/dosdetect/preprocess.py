"""Encode features and keep the top-k by |correlation| (fit on train only)."""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from dosdetect.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


class Preprocessor:
    def __init__(self, n_features=8):
        self.n_features = n_features
        # Categories missing from training become -1 at test time.
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.correlations = None
        self.selected_features = None

    def fit(self, df):
        X = self._encode(df, fit=True)
        # Constant columns give NaN correlation; treat as 0.
        corr = X.corrwith(df["label"]).abs().fillna(0)
        self.correlations = corr.sort_values(ascending=False)
        self.selected_features = list(self.correlations.index[: self.n_features])
        return self

    def transform(self, df):
        X = self._encode(df, fit=False)
        return X[self.selected_features], df["label"]

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def _encode(self, df, fit):
        X = df[FEATURE_COLUMNS].copy()
        cats = X[CATEGORICAL_COLUMNS]
        encoded = self.encoder.fit_transform(cats) if fit else self.encoder.transform(cats)
        X[CATEGORICAL_COLUMNS] = pd.DataFrame(encoded, columns=CATEGORICAL_COLUMNS, index=X.index)
        return X.astype(float)
