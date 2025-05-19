#!/usr/bin/env python3
import os
import pandas as pd
import h2o

# Configuration 
MODEL_PATH          = '/Users/marinorfolk/Desktop/mlops-2/models/h2o_models/StackedEnsemble_AllModels_1_AutoML_1_20250518_185835'
TEST_X_PATH         = '/Users/marinorfolk/Desktop/mlops-2/data/processed/X_test.csv'
TEST_Y_PATH         = '/Users/marinorfolk/Desktop/mlops-2/data/processed/y_test.csv'
TARGET_COLUMN_NAME  = 'diagnosis'
POSITIVE_CLASS      = '1'

MODEL_OUTPUT_DIR  = '/Users/marinorfolk/Desktop/mlops-2/models/h2o_models/'
BEST_PATH_FILE    = os.path.join(MODEL_OUTPUT_DIR, "best_model_path.txt")

# Read the best model path from the file
with open(BEST_PATH_FILE) as f:
    model_path = f.read().strip()

# Load best model 
h2o.init(nthreads=-1, max_mem_size="8G")
model = h2o.load_model(model_path)
print(f"Loaded leader: {model.model_id}")

# Load and prepare test data
X_test = pd.read_csv(TEST_X_PATH)
y_test = pd.read_csv(TEST_Y_PATH)
if len(y_test.columns) == 1 and y_test.columns[0] != TARGET_COLUMN_NAME:
    y_test.columns = [TARGET_COLUMN_NAME]

df_test = pd.concat([X_test, y_test], axis=1)
hf_test = h2o.H2OFrame(df_test)
hf_test[TARGET_COLUMN_NAME] = hf_test[TARGET_COLUMN_NAME].asfactor()

# Scores 
perf_test = model.model_performance(hf_test)

print("\n=== Test Set Performance ===")
print(f"AUC:     {perf_test.auc():.4f}")
print(f"AUCPR:   {perf_test.aucpr():.4f}")
print(f"Logloss: {perf_test.logloss():.4f}\n")

cm = perf_test.confusion_matrix()
print("Confusion Matrix:")
print(cm)

cm_df = cm.table.as_data_frame().rename(columns={cm.table.as_data_frame().columns[0]: 'actual'})
cm_df = cm_df[cm_df['actual'] != 'Total']

neg = cm_df[cm_df['actual'] != POSITIVE_CLASS].iloc[0]
pos = cm_df[cm_df['actual'] == POSITIVE_CLASS].iloc[0]

TN, FP = int(neg['0']), int(neg['1'])
FN, TP = int(pos['0']), int(pos['1'])

accuracy  = (TP + TN) / (TP + TN + FP + FN)
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

print(f"\nDerived Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1 Score:  {2 * (precision * recall) / (precision + recall):.4f}")

# Shut down H2O cluster
h2o.cluster().shutdown(prompt=False)
