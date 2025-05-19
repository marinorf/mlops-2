import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import recall_score

import ray # needed to do this to manually shut it down at the end, since it stalled 

# Configuration 
DATA_PATH_X = '/Users/marinorfolk/Desktop/mlops-2/data/processed/X_train.csv'
DATA_PATH_Y = '/Users/marinorfolk/Desktop/mlops-2/data/processed/y_train.csv'
MODEL_OUTPUT_DIR = '/Users/marinorfolk/Desktop/mlops-2/models/autogluon_models/'
TARGET_COLUMN_NAME = 'diagnosis'
POSITIVE_CLASS = 1
MAX_RUNTIME_SECS = 3600
SEED = 42

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Load and Prepare Data 
X_train = pd.read_csv(DATA_PATH_X)
y_train = pd.read_csv(DATA_PATH_Y)
if len(y_train.columns) == 1 and y_train.columns[0] != TARGET_COLUMN_NAME:
    y_train.columns = [TARGET_COLUMN_NAME]
elif TARGET_COLUMN_NAME not in y_train.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN_NAME}' not found in {DATA_PATH_Y}")
full_data = pd.concat([X_train, y_train[TARGET_COLUMN_NAME]], axis=1)

# Reproducible Split
random.seed(SEED)
np.random.seed(SEED)
train_data, tuning_data = train_test_split(
    full_data,
    test_size=0.1,
    stratify=full_data[TARGET_COLUMN_NAME],
    random_state=SEED
)

# Train 
predictor = TabularPredictor(
    label=TARGET_COLUMN_NAME,
    path=MODEL_OUTPUT_DIR,
    problem_type='binary',
    eval_metric='roc_auc',
    verbosity=2
).fit(
    train_data=train_data,
    tuning_data=tuning_data,
    time_limit=MAX_RUNTIME_SECS,
    presets='best_quality',
    use_bag_holdout=True,
)

print("AutoGluon training complete.\n")

# Leaderboard & Recall 
print("Model leaderboard:")
ld_bd = predictor.leaderboard(silent=True)

#print the 10 best models 
print(ld_bd.head(10))
best_model = predictor.get_model_best()
print(f"\nBest model (used by default in predict()): {best_model}")
print(f"\nHyperparameters for '{best_model}':")
print(predictor.model_hyperparameters(best_model))

# Save Predictor
predictor.save(silent=True)
print(f"\nPredictor saved to: {MODEL_OUTPUT_DIR}")

# Shutdown Ray - maybe not needed, but I got some problems with termination wihtout it 
if ray.is_initialized():
    ray.shutdown()   
