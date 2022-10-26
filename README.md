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
    python main_optuna.py -d True -n 5

```-d``` allows to drop certain features and ```-n``` is the number of trials.

## MLFlow dashboard

To view all the experiments:

    mlflow ui
