import optuna
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

from optuna.integration.mlflow import MLflowCallback
from sklearn import model_selection

from models import models_list
from src.preprocess.preprocess import pipeline, preprocess_x, preprocess_y


mlflc = MLflowCallback(metric_name="accuracy")


@mlflc.track_in_mlflow()
def objective(trial):

    data_train = pd.read_csv("../../data/train_airbnb_berlin.xls")

    Y = preprocess_y(data_train)
    X = pd.read_csv("../../data/X_train.csv")

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

    score = model_selection.cross_val_score(classifier_obj, X, Y, scoring="r2", n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(study_name="my_study", direction="maximize")
study.optimize(objective, n_trials=200, callbacks=[mlflc])


print(study.best_params)  # Show the best value.
