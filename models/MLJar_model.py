import pandas as pd 
import os
import json
import shutil

from supervised.automl import AutoML

# import data from file
RESULTS_PATH = "models/AutoML_results"

if os.path.exists(RESULTS_PATH):
    shutil.rmtree(RESULTS_PATH)

X_train = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/processed/X_train.csv')
y_train = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/processed/y_train.csv')

# Model training
a = AutoML(results_path=RESULTS_PATH, mode = "Perform", total_time_limit = 300,
           n_jobs = -1, ml_task = "binary_classification",
           eval_metric = 'f1')

a.fit(X_train, y_train)

a.report()