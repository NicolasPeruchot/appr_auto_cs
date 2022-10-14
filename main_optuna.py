from dataclasses import dataclass

import optuna
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

from optuna.integration.mlflow import MLflowCallback
from sklearn import datasets, metrics, model_selection

from src.models import models_list


mlflc = MLflowCallback(metric_name="accuracy")


@mlflc.track_in_mlflow()
def objective(trial):
    diabete = datasets.load_diabetes()
    x, y = diabete.data, diabete.target

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

        if param["type"] == "categorical":
            current_params[param["optuna_params"]["name"]] = trial.suggest_categorical(
                param["optuna_params"]["name"],
                param["optuna_params"]["cat"],
            )

    classifier_obj = models_list[classifier_name]["model"](**current_params)

    score = model_selection.cross_val_score(classifier_obj, x, y, scoring="r2", n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(study_name="my_study", direction="maximize")
study.optimize(objective, n_trials=20, callbacks=[mlflc])


print(study.best_params)  # Show the best value.
