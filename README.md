# Berlin AirBnb Price Prediction



## Code organisation
    data => datasets
    │
    src
    ├── preprocess
    │   └── preprocess.py => preprocessing methods and pipeline
    │   └── utils.py => misc methods
    └── training
        └── main_optuna.py => training script with hyperparameters search
        └── models.py => list of all models and hyperparameters

## Installation

    make install

## Training

To launch a training:

    cd src/training
    python main_optuna.py -n 5

```-n``` is the number of trials. It is possible to use the flag ```--drop``` to drop certain features.

## MLFlow dashboard

To view all the experiments:

    mlflow ui
