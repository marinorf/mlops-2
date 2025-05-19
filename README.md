# 🤖 ML Ops 2 Workflow for Breast Cancer Wisconsin

This repository implements a full ML workflow for the Breast Cancer Wisconsin dataset, developed as part of the ML Operations course at EWHA University. All steps from data loading through model training and evaluation are managed through the Makefile.

---

## ⚙️ Workflow Overview

1. **Data Management**  
2. **Preprocessing**  
3. **Model Training**  
4. **Model Evaluation**  
5. **Dependency Management**

---

### 1. Data Management

#### Download Data  
Fetch the Breast Cancer Wisconsin dataset from Kaggle and save it as `data/raw.csv`.

<blockquote>
<pre>
make download-data
</pre>
</blockquote>

> **Prerequisite:** You must have a valid `kaggle.json` in `~/.kaggle/` (or set `KAGGLE_CONFIG_DIR`) for Kaggle API authentication.

---

### 2. Preprocessing

#### Preprocess Raw Data  
Clean the raw CSV, engineer features, and split into train/test sets:
- `data/X_train.csv`
- `data/y_train.csv`
- `data/X_test.csv`
- `data/y_test.csv`

<blockquote>
<pre>
make preprocess-data
</pre>
</blockquote>

---

### 3. Model Training

#### Train MLJAR AutoML Model  
Launch an AutoML run with MLJAR‑Supervised, storing results in `models/AutoML_results/`.

<blockquote>
<pre>
make train-MLJar
</pre>
</blockquote>

#### Train H2O AutoML Model  
Launch an AutoML run with H20, storing results in `models/h2o_models/`.

<blockquote>
<pre>
make train-h2o
</pre>
</blockquote>

#### Train AutoGluon Model
Launch an AutoML run with AutoGluon, storing results in `models/autogluon_models/`.

<blockquote>
<pre>
make train-autogluon
</pre>
</blockquote>

---

### 4. Model Evaluation

#### Evaluate All Models  
Compute accuracy, precision, recall, F1, confusion matrices, and output predictions under `evaluate/`.

<blockquote>
<pre>
make evaluate-models
</pre>
</blockquote>

---

### 5. Dependency Management

#### Install Python Packages  
Install all required libraries as pinned in `requirements.txt`. We have two `requirements.txt` files in this project, one explicitly called `requirements_ag.txt` for the AutoGluon environment.

<blockquote>
<pre>
make install
</pre>
</blockquote>

---

## 📂 Repository Structure
```
.
├── Makefile                     # Automates the workflow
├── requirements.txt             # Python dependencies
├── requirements_ag.txt          # Python dependencies for AutoGluon environment 
├── data/                        # Raw & preprocessed datasets
│   ├── raw.csv
│   ├── processed/
│       ├── X_train.csv
│       ├── y_train.csv
│       ├── X_test.csv
│       └── y_test.csv
├── models/                      # Model python files and saved models
│   ├── autogluon.py
│   ├── MLJar_model.py
│   └── h2o_model.py        
├── scripts/                     # Pipeline scripts
│   └── preprocess.py            # Data cleaning & train/test seperation
├── evaluate/                    # Model evaluation scripts
│   ├── evaluate_autogluon.py
│   ├── evaluate_h20.py
│   └── evaluate_MLJar.py
```

### 📝 Notes
Testing - tests of functions are not yet implemented.

Kaggle Authentication - Ensure your Kaggle API token (kaggle.json) is in ~/.kaggle/.

