import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
from supervised.automl import AutoML

# Load the preprocessed data
X_test = pd.read_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/X_test.csv')
y_test = pd.read_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/y_test.csv')['diagnosis']

# Load the best model
RESULTS_PATH = "models/AutoML_results"
a = AutoML(results_path=RESULTS_PATH, mode="Predict", total_time_limit=300,
           n_jobs=1, ml_task="binary_classification",
           eval_metric='f1')

# Model evaluation
y_pred = a.predict(X_test)

print(f"Model score {accuracy_score(y_test, y_pred)} on set")

print("Confusion matrix: ", confusion_matrix(y_test, y_pred))

print("Classification report: ", classification_report(y_test, y_pred))

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('/Users/marinorfolk/Desktop/mlops-2/data/MLJar_predictions.csv', index=False)

# Load the leaderboard
RESULTS_PATH = "models/AutoML_results"
NUMBER_OF_MODELS = 3

lb_csv = os.path.join(RESULTS_PATH, "leaderboard.csv")
df = pd.read_csv(lb_csv)

df_sorted = df.sort_values("metric_value", ascending=False)

top = df_sorted.head(NUMBER_OF_MODELS)

print(f"Top {NUMBER_OF_MODELS} models by validation score:")
for i, model in top.iterrows():
    print(f" {i}. {model['name']:20s} with score: {model['metric_value']:.4f}")
    