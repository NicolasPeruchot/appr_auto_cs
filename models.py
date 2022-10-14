from sklearn import ensemble, svm


models_list = {
    "svm": {
        "model": svm.SVC,
        "hyperparams": [
            {
                "type": "float",
                "optuna_params": {"name": "C", "low": 1e-10, "high": 1e10, "log": True},
            }
        ],
    },
    "random_forest": {
        "model": ensemble.RandomForestClassifier,
        "hyperparams": [
            {
                "type": "int",
                "optuna_params": {"name": "n_estimators", "low": 10, "high": 1000, "log": True},
            },
            {
                "type": "int",
                "optuna_params": {"name": "max_depth", "low": 2, "high": 32, "log": True},
            },
        ],
    },
}
