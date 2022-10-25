from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocess.utils import drop_useless, incomplete_columns


def preprocess_x(data):
    """Clean the data"""
    data = drop_useless(data)

    data = data.replace("*", np.nan)

    listfeatures = incomplete_columns(data, to_print=False)
    data = data.drop(columns=[x for x in listfeatures.keys() if listfeatures[x]["pct"] > 45])

    data["Property Type"] = np.where(
        data["Property Type"].isna(), "Apartment", data["Property Type"]
    )

    obj = set(data.select_dtypes(["object"]).columns)
    na = set(data.columns[data.isna().any()].tolist())
    data = data.astype({x: "float64" for x in obj.intersection(na)})
    return data


class CustomOneHotEncoder(OneHotEncoder):
    def __init__(self, categories="auto"):
        super().__init__(categories=categories)

    def fit(self, X, y=None):
        X = preprocess_x(X)
        self.features_to_encode = list(X.select_dtypes(["object"]).columns)
        return super().fit(X[self.features_to_encode], y)

    def transform(self, X, y=None):
        X = preprocess_x(X)
        one_hot_encoded = pd.DataFrame(
            super().transform(X[self.features_to_encode]).toarray(),
            columns=self.get_feature_names_out(),
        )

        return pd.concat(
            [
                X.drop(columns=self.features_to_encode).reset_index(drop=True),
                one_hot_encoded.reset_index(drop=True),
            ],
            axis=1,
            ignore_index=True,
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class CustomIterativeImputer(IterativeImputer):
    def __init__(self, sample_posterior=True, random_state=0):
        super().__init__(sample_posterior=sample_posterior, random_state=random_state)

    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X, y), columns=X.columns)

    def fit_transform(self, X, y=None):
        return pd.DataFrame(super().fit_transform(X, y), columns=X.columns)


pipeline = Pipeline(
    [
        ("one_hot", CustomOneHotEncoder()),
        ("iterative", IterativeImputer()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.80)),
    ]
)
