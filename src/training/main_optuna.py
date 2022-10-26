import argparse

import optuna
import pandas as pd

from models import models_list
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocess.preprocess import pipeline, preprocess_data


mlflc = MLflowCallback(metric_name="accuracy")

data = pd.read_csv("../../data/train_airbnb_berlin.xls")


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--drop-all-ratings", required=True, type=bool)
parser.add_argument("-n", "--n-trials", required=True, type=int)
args = vars(parser.parse_args())

drop_all_ratings = args["drop_all_ratings"]
n_trials = args["n_trials"]


data = preprocess_data(data, drop_all_ratings=drop_all_ratings)

Y = data["Price"].values
X = data.drop(columns=["Price"])


X_train, X_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
)


X_train = pipeline.fit_transform(
    X_train,
    one_hot__drop_all_ratings=drop_all_ratings,
)

X_val = pipeline.transform(X_val)


@mlflc.track_in_mlflow()
def objective(trial):
    trial.suggest_categorical("drop_all_ratings", [drop_all_ratings])
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
    score = cross_val_score(
        classifier_obj,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=5,
        verbose=1,
    )
    accuracy = score.mean()
    return accuracy


study = optuna.create_study(study_name="study", direction="maximize")
study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])


print(study.best_params)  # Show the best value.

best = study.best_params
model_name = best["classifier"]
del best["classifier"]
del best["drop_all_ratings"]
model = models_list[model_name]["model"](**best)
model.fit(X_train, y_train)

print(f"Score de validation: {mean_squared_error(model.predict(X_val),y_val)}")
