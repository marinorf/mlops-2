SHELL := /bin/bash

CONDA_BASE := $(shell conda info --base)
CONDA_INIT := $(CONDA_BASE)/etc/profile.d/conda.sh

.PHONY: train-autogloun

#Data management 
download-data:
	@mkdir -p data
	# download & unzip the CSV from Kaggle into data/
	kaggle datasets download \
	  -d uciml/breast-cancer-wisconsin-data \
	  -f data.csv \
	  -p data/ \
	  --unzip
	# rename to raw.csv
	mv data/data.csv data/raw.csv 

preprocess-data:
	@mkdir -p data/processed
	# run the preprocessing script
	python3 scripts/preprocess.py

# Model training
train-h2o:
	@mkdir -p models
	# run the training script
	python3 models/h2o_model.py

train-MLJar:
	@mkdir -p models
	# run the training script
	python3 models/MLJAR_model.py

train-autogluon:
	@mkdir -p models
	# run the training script
	@/Users/marinorfolk/miniconda3/bin/conda run -n ag-env \
	      --no-capture-output python3 models/autogluon.py

# Model evaluation
evaluate-autogluon:
	@mkdir -p evaluate
	# run the evaluation script
	@/Users/marinorfolk/miniconda3/bin/conda run -n ag-env \
	      --no-capture-output python3 evaluate/evaluate_autogluon.py

evaluate-h20:
	@mkdir -p evaluate
	# run the evaluation script
	python3 evaluate/evaluate_h2o.py

evaluate-MLJar:
	@mkdir -p evaluate
	# run the evaluation script
	python3 evaluate/evaluate_MLJar.py

