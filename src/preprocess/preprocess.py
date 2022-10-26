import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocess.utils import (
    drop_missing_too_high,
    drop_missing_too_low,
    drop_ratings,
    drop_useless,
)


def preprocess_data(data, drop_all_ratings=False):
    """Clean the data.
    drop_all_ratings allows to choose if the ratings are kept."""
    data = data.dropna(subset=["Price"]).reset_index(drop=True)
    data = drop_useless(data)

    data = data.replace("*", np.nan)

    data = drop_missing_too_high(data)

    data["Property Type"] = np.where(
        data["Property Type"].isna(), "Apartment", data["Property Type"]
    )
    data = drop_missing_too_low(data)

    obj = set(data.select_dtypes(["object"]).columns)
    na = set(data.columns[data.isna().any()].tolist())
    data = data.astype({x: "float64" for x in obj.intersection(na)})

    if drop_all_ratings:
        data = drop_ratings(data)
    return data


class CustomOneHotEncoder(OneHotEncoder):
    def __init__(self, categories="auto", drop_all_ratings=False):
        self.drop_all_ratings = drop_all_ratings
        super().__init__(categories=categories)

    def fit(self, X, y=None, drop_all_ratings=None):
        if drop_all_ratings == None:
            drop_all_ratings = self.drop_all_ratings
        self.features_to_encode = [
            "Neighborhood Group",
            "Property Type",
            "Room Type",
            "Guests Included",
            "Instant Bookable",
            "Business Travel Ready",
        ]
        return super().fit(X[self.features_to_encode], y)

    def transform(self, X, y=None, drop_all_ratings=None):
        if drop_all_ratings == None:
            drop_all_ratings = self.drop_all_ratings
        one_hot_encoded = pd.DataFrame(
            super().transform(X[self.features_to_encode]).toarray(),
            columns=self.get_feature_names_out(),
        )
        X = X.drop(columns=self.features_to_encode).reset_index(drop=True)
        one_hot_encoded = one_hot_encoded.reset_index(drop=True)

        concat = pd.concat(
            [
                X,
                one_hot_encoded,
            ],
            axis=1,
        )

        return concat

    def fit_transform(self, X, y=None, drop_all_ratings=None):
        if drop_all_ratings == None:
            drop_all_ratings = self.drop_all_ratings
        self.fit(X, y, drop_all_ratings=drop_all_ratings)
        return self.transform(X, y, drop_all_ratings=drop_all_ratings)


pipeline = Pipeline(
    [
        ("one_hot", CustomOneHotEncoder()),
        ("iterative", IterativeImputer()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.80)),
    ]
)
