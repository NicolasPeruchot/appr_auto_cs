from sklearn import ensemble, tree
from xgboost import XGBRegressor


models_list = {
    "DecisionTreeRegressor": {
        "model": tree.DecisionTreeRegressor,
        "hyperparams": [  # bonne idée d'inclure la profondeurmax dans les hyperparametres non ?
            {
                "type": "categorical",
                "optuna_params": {
                    "name": "criterion",
                    "cat": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
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
                    "high": 100,
                },
            },
            # {
            #     "type": "int",
            #     "optuna_params": {
            #         "name": "max_features",
            #         "low": 1,
            #         "high": 40,
            #     },
            # },
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
                    "high": 100,
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
            # {
            #     "type": "categorical",
            #     "optuna_params": {
            #         "name": "loss",
            #         "cat": [
            #             "squared_error",
            #             "absolute_error",
            #             "huber",
            #             "quantile",
            #         ],
            #     },
            # },
            {
                "type": "int",
                "optuna_params": {
                    "name": "n_estimators",
                    "low": 1,
                    "high": 100,
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
            # {
            #     "type": "int",
            #     "optuna_params": {
            #         "name": "n_estimators",
            #         "low": 1,
            #         "high": 100,
            #     },
            # },
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
                    "high": 100,
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
