import optuna
import pandas as pd

from models import models_list
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from src.preprocess.preprocess import pipeline
from src.preprocess.utils import drop_ratings


mlflc = MLflowCallback(metric_name="accuracy")

data = pd.read_csv("../../data/train_airbnb_berlin.xls")
data = data.dropna(subset=["Price"]).reset_index(drop=True)

# valeur à switcher dans notre main
# si interrupteur = True : on supprime toutes les lignes des ratings qui ont des NA
# si interrupteur = False : on fait des regressions stochastiques sur ces lignes
drop_all_ratings = False

if drop_all_ratings:
    Y = drop_ratings(data)["Price"]
else:
    Y = data["Price"]

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

    classifier_name = trial.suggest_categorical("classifier", models_list.keys())
    print(classifier_name)

    current_params = {}
    for param in models_list[classifier_name]["hyperparams"]:
        print(param)
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


study = optuna.create_study(study_name="my_study", direction="maximize")
study.optimize(objective, n_trials=20, callbacks=[mlflc])


print(study.best_params)  # Show the best value.

best = study.best_params
model_name = best["classifier"]
del best["classifier"]
model = models_list[model_name]["model"](**best)
model.fit(X_train, y_train)

print(f"Score de validation: {mean_squared_error(model.predict(X_val),y_val)}")
