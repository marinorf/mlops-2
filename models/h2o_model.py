import os
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

# Configuration
DATA_PATH_X        = '/Users/marinorfolk/Desktop/mlops-2/data/processed/X_train.csv'
DATA_PATH_Y        = '/Users/marinorfolk/Desktop/mlops-2/data/processed/y_train.csv'
MODEL_OUTPUT_DIR   = '/Users/marinorfolk/Desktop/mlops-2/models/h2o_models/'
TARGET_COLUMN_NAME = 'diagnosis'
MAX_MODELS         = 20
MAX_RUNTIME_SECS   = 3600
CV_FOLDS           = 5
SEED               = 42

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Start H2O 
h2o.init(nthreads=-1, max_mem_size="8G")

# Load & Prepare Data
X = pd.read_csv(DATA_PATH_X)
y = pd.read_csv(DATA_PATH_Y)
if len(y.columns) == 1 and y.columns[0] != TARGET_COLUMN_NAME:
    y.columns = [TARGET_COLUMN_NAME]
df = pd.concat([X, y], axis=1)

hf = h2o.H2OFrame(df)
hf[TARGET_COLUMN_NAME] = hf[TARGET_COLUMN_NAME].asfactor()

# Train AutoML
aml = H2OAutoML(
    max_models=MAX_MODELS,
    max_runtime_secs=MAX_RUNTIME_SECS,
    seed=SEED,
    nfolds=CV_FOLDS,
    balance_classes=True,
)
aml.train(x=X.columns.tolist(), y=TARGET_COLUMN_NAME, training_frame=hf)

# Trying to sort leaderboard by F1 to compare with MLJar on right metric and then use that model
lb_df = aml.leaderboard.as_data_frame(use_pandas=True)

def compute_max_f1(model_id):
    model = h2o.get_model(model_id)
    perf = model.model_performance(xval=True)
    mjson = perf._metric_json
    f1_list = mjson['thresholds_and_metric_scores']['f1']
    
    return max(f1_list)

lb_df['max_f1'] = lb_df['model_id'].apply(compute_max_f1)
lb_df = lb_df.sort_values('max_f1', ascending=False).reset_index(drop=True)

# Display the leaderboard
print("\nAutoML Leaderboard (sorted by max F1):")
print(lb_df[['model_id','max_f1']].head(10))

best_model_id = lb_df.loc[0, 'model_id']
best = h2o.get_model(best_model_id)
print(f"\nBest model by F1: {best_model_id}")

# Report Standard Metrics on Training Data 
perf = best.model_performance(train=True)
print("\nPerformance on training data:")
print(f"  AUC:   {perf.auc():.4f}")
print(f"  AUCPR: {perf.aucpr():.4f}")
print(f"  Logloss: {perf.logloss():.4f}")
print("\n  Confusion Matrix:")
print(perf.confusion_matrix())

# Save the Leader Model
model_path = h2o.save_model(model=best, path=MODEL_OUTPUT_DIR, force=True)
print(f"\nSaved H2O binary model to: {model_path}")

with open(os.path.join(MODEL_OUTPUT_DIR, "best_model_path.txt"), "w") as f:
    f.write(model_path)

try:
    mojo_path = best.download_mojo(path=MODEL_OUTPUT_DIR, get_genmodel_jar=True)
    print(f"Saved MOJO to: {mojo_path}")
except Exception:
    print("MOJO export not supported for this model.")
    
# Shutdown H2O 
h2o.cluster().shutdown(prompt=False)
