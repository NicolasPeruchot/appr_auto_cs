import optuna
import pandas as pd
import sklearn.model_selection
import sklearn.svm

from models import models_list
from optuna.integration.mlflow import MLflowCallback
from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split

from src.preprocess.preprocess import pipeline


mlflc = MLflowCallback(metric_name="accuracy")

data = pd.read_csv("../../data/train_airbnb_berlin.xls")
data = data.dropna(subset=["Price"]).reset_index(drop=True)


Y = data["Price"]
X = data.drop(columns=["Price"])


X_train, X_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
)

X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)


@mlflc.track_in_mlflow()
def objective(trial):

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
    score = model_selection.cross_val_score(
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


study = optuna.create_study(study_name="my_study", direction="maximize")
study.optimize(objective, n_trials=20, callbacks=[mlflc])


print(study.best_params)  # Show the best value.

best = study.best_params
model_name = best["classifier"]
del best["classifier"]
model = models_list[model_name]["model"](**best)
model.fit(X_train, y_train)

print(f"Score de validation: {metrics.mean_squared_error(model.predict(X_val),y_val)}")
