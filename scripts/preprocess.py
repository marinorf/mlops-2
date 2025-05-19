import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
import os
import cv2

from supervised.automl import AutoML

# import data from file
# TO DO change to from curl from kaggle 
PATH = "/Users/marinorfolk/Desktop/mlops-2/data/raw.csv" 

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocessing(df):
    # Convert categorical variables to numerical
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    # Drop unnecessary columns
    df.drop(columns=['id'], inplace=True)
    df.drop(columns=['Unnamed: 32'], inplace=True)
    return df
    
df = load_data(PATH)
df = preprocessing(df)

# check for missing values
print(df.isnull().sum())
# check for duplicates
print(df.duplicated().sum())

# split into test and train
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# write to file 
X_train.to_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/X_train.csv', index=False)
X_test.to_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/X_test.csv', index=False)
y_train.to_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/y_train.csv', index=False)
y_test.to_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/y_test.csv', index=False)