from sklearn import ensemble, tree
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


models_list = {
    "KNN": {
        "model": KNeighborsRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {
                    "name": "n_neighbors",
                    "low": 2,
                    "high": 200,
                },
            },
        ],
    },
    "DecisionTreeRegressor": {
        "model": tree.DecisionTreeRegressor,
        "hyperparams": [
            {
                "type": "categorical",
                "optuna_params": {
                    "name": "criterion",
                    "cat": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                    ],
                },
            },
            {
                "type": "categorical",
                "optuna_params": {
                    "name": "splitter",
                    "cat": [
                        "best",
                        "random",
                    ],
                },
            },
        ],
    },
    "BaggingRegressor": {
        "model": ensemble.BaggingRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {
                    "name": "n_estimators",
                    "low": 10,
                    "high": 500,
                },
            },
            {
                "type": "int",
                "optuna_params": {
                    "name": "max_samples",
                    "low": 1,
                    "high": 10,
                },
            },
            {
                "type": "float",
                "optuna_params": {
                    "name": "max_features",
                    "low": 0.1,
                    "high": 1,
                },
            },
        ],
    },
    "RandomForestRegressor": {
        "model": ensemble.RandomForestRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {
                    "name": "n_estimators",
                    "low": 10,
                    "high": 500,
                },
            },
            {
                "type": "int",
                "optuna_params": {
                    "name": "max_depth",
                    "low": 2,
                    "high": 10,
                },
            },
        ],
    },
    "GradientBoostingRegression": {
        "model": ensemble.GradientBoostingRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {
                    "name": "n_estimators",
                    "low": 1,
                    "high": 500,
                },
            },
            {
                "type": "float",
                "optuna_params": {
                    "name": "learning_rate",
                    "low": 0,
                    "high": 0.5,
                },
            },
        ],
    },
    "AdaBoostRegressor": {
        "model": ensemble.AdaBoostRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {
                    "name": "n_estimators",
                    "low": 1,
                    "high": 500,
                },
            },
            {
                "type": "float",
                "optuna_params": {
                    "name": "learning_rate",
                    "low": 0,
                    "high": 0.5,
                },
            },
            {
                "type": "categorical",
                "optuna_params": {
                    "name": "loss",
                    "cat": [
                        "linear",
                        "square",
                        "exponential",
                    ],
                },
            },
        ],
    },
    "XGBRegressor": {
        "model": XGBRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {
                    "name": "n_estimators",
                    "low": 10,
                    "high": 500,
                },
            },
            {
                "type": "int",
                "optuna_params": {
                    "name": "max_depth",
                    "low": 1,
                    "high": 10,
                },
            },
            {
                "type": "float",
                "optuna_params": {
                    "name": "eta",
                    "low": 0.01,
                    "high": 0.3,
                },
            },
        ],
    },
}
