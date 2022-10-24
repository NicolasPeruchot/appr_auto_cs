from sklearn import ensemble, tree
from xgboost import XGBRegressor


models_list = {
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
        "hyperparam": [
            {
                "type": "int",
                "optuna_params": {"name": "n_estimators", "low": 10, "high": 1000, "log": True, },
            },
            {
                "type": "int",
                "optuna_params": {"name": "max_samples", "low": 0.5, "high": 1, "log": True, },
            },
            {
                "type": "int",
                "optuna_params": {"name": "max_features", "low": 0.1, "high": 1, "log": True, },
            },
            {
                "type": "categorical",
                "optuna_params": {
                    "name": "base_estimator",
                    "cat": [
                        "DecisionTreeRegressor",
                        "SVR",
                        "AdaBoostRegressor",
                    ],
                },
            },
        ],
    },

    "RandomForestRegressor": {
        "model": ensemble.RandomForestRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {"name": "n_estimators", "low": 10, "high": 10000, "log": True, },
            },
            {
                "type": "int",
                "optuna_params": {"name": "max_depth", "low": 2, "high": 32, "log": True, },
            },
        ],
    },

    # "GradientBoostingRegression": {
    #     "model": ensemble.GradientBoostingRegressor,
    #     "hyperparam": [
    #         # TODO
    #     ],
    # },

    # "AdaBoostRegressor": {
    #     "model": ensemble.AdaBoostRegressor,
    #     "hyperparam": [
    #         # TODO
    #     ],
    # },

    "XGBRegressor": {
        "model": XGBRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {"name": "n_estimators", "low": 10, "high": 10000, "log": True},
            },
            {
                "type": "int",
                "optuna_params": {"name": "max_depth", "low": 1, "high": 10, "log": True},
            },
            {
                "type": "float",
                "optuna_params": {"name": "eta", "low": 0.01, "high": 0.3, "log": True},
            },
        ],
    },
}
