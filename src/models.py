from sklearn import ensemble, svm, tree


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
    "RandomForestRegressor": {
        "model": ensemble.RandomForestRegressor,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {"name": "n_estimators", "low": 10, "high": 10000, "log": True},
            },
            {
                "type": "int",
                "optuna_params": {"name": "max_depth", "low": 2, "high": 32, "log": True},
            },
        ],
    },
}
