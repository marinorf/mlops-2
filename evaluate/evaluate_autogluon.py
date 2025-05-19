# evaluate_autogluon.py
import os
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load test data
X_test = pd.read_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/X_test.csv')
y_test = pd.read_csv('/Users/marinorfolk/Desktop/mlops-2/data/processed/y_test.csv')['diagnosis']

# Load the trained AutoGluon predictor
MODEL_OUTPUT_DIR = '/Users/marinorfolk/Desktop/mlops-2/models/autogluon_models/'
predictor = TabularPredictor.load(MODEL_OUTPUT_DIR)

# Identify the single 'best' model used by predict() by default
best_model = predictor.model_best
print(f"\nBest model (used by default in predict()): {best_model}")

# Show that model’s hyperparameters
print(f"\nHyperparameters for '{best_model}':")
print(predictor.model_hyperparameters(best_model))

# Build the test‐set 
test_data = pd.concat([X_test, y_test.rename('diagnosis')], axis=1)

metrics = predictor.evaluate(
    test_data,
    silent=True         
)

print(metrics)

# Predict & compute confusion matrix
y_pred = predictor.predict(X_test, model=best_model)
print(f"\nAccuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the predictions
preds_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
out_path = '/Users/marinorfolk/Desktop/mlops-2/data/autogluon_predictions.csv'
preds_df.to_csv(out_path, index=False)
print(f"\nPredictions written to: {out_path}")
