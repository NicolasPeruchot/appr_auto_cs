from dataclasses import dataclass

import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

from mlflow import sklearn
from optuna.integration.mlflow import MLflowCallback
from sklearn import datasets, ensemble, model_selection, svm

from models import models_list


mlflc = MLflowCallback(metric_name="accuracy")


@mlflc.track_in_mlflow()
def objective(trial):
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", models_list.keys())

    current_params = {}
    for param in models_list[classifier_name]["hyperparams"]:
        if param["type"] == "int":
            current_params[param["optuna_params"]["name"]] = trial.suggest_int(
                **param["optuna_params"]
            )
        if param["type"] == "float":
            current_params[param["optuna_params"]["name"]] = trial.suggest_float(
                **param["optuna_params"]
            )

    classifier_obj = models_list[classifier_name]["model"](**current_params)

    score = model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(study_name="my_study", direction="maximize")
study.optimize(objective, n_trials=10, callbacks=[mlflc])


print(study.best_params)  # Show the best value.
