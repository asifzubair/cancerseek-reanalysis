# CancerSeek-reanalysis

This project is a extension of the machine learning modelling from the paper "Detection and localization of surgically resectable cancers with a multi-analyte blood test" by Cohen et al. (2018). The goal is to predict cancer based on proteomic and genetic (mutation) data from patient blood samples.

## Project Overview

This repository contains a complete machine learning pipeline to:
- Preprocess proteomic and mutation data from the supplementary materials of the paper.
- Train a baseline logistic regression model.
- Train an autoencoder on healthy patient data to generate a "reconstruction error" feature, which helps in identifying anomalous (cancerous) samples.
- Train a deep learning model (`CancerPredictor`) that uses a combination of numerical features, mutation data, and the autoencoder's reconstruction error to classify samples.
- Perform hyperparameter tuning using Optuna.
- Evaluate the models using 10-fold stratified cross-validation.

## Dataset

The dataset used in this project is from the supplementary materials of the following publication:

Cohen, J. D., Li, L., Wang, Y., Thoburn, C., Afsari, B., Danilova, L., ... & Hruban, R. H. (2018). Detection and localization of surgically resectable cancers with a multi-analyte blood test. *Science*, *359*(6378), 926-930.

The data is provided in the `data/NIHMS982921-supplement-Tables_S1_to_S11.xlsx` file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/asifzubair/cancerseek-reanalysis
   cd cancerseek-reanalysis
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Hyperparameter Tuning

To find the best hyperparameters for the `CancerPredictor` model, run the following command:

```bash
python -c "from src.train import tune_hyperparameters; tune_hyperparameters()"
```

This will run Optuna to find the best hyperparameters and print them to the console. The best parameters can then be updated in `config.py`.

### Cross-Validation

To run 10-fold cross-validation for the main `CancerPredictor` model, run:

```bash
python -c "from src.train import run_cross_validation; run_cross_validation()"
```

To run 10-fold cross-validation for the baseline logistic regression model, run:

```bash
python -c "from src.train import run_baseline_cross_validation; run_baseline_cross_validation()"
```

These scripts will print the average validation score and save a dataframe with the out-of-fold predictions.

## Project Structure

```
├── data/
│   └── NIHMS982921-supplement-Tables_S1_to_S11.xlsx  # Raw data
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── etl.py                      # Data loading and preprocessing
│   ├── models.py                   # PyTorch Lightning models
│   ├── train.py                    # Training, tuning, and cross-validation logic
│   └── utils.py
├── config.py                       # Configuration variables
└── README.md                       
```

## Modeling

### 1. Autoencoder

An autoencoder is trained exclusively on the protein data of healthy individuals. The reconstruction error of this model is then used as a feature for the downstream classifier. The idea is that samples from cancer patients will have a higher reconstruction error, as the autoencoder has not been trained on them.

### 2. Logistic Regression Baseline

A simple logistic regression model is used as a baseline. It is trained on a subset of the protein features and the `omega_score`.

### 3. CancerPredictor

This is the main model, a neural network built with PyTorch Lightning. It takes as input:
- A set of selected numerical features (proteins and mutation features).
- An embedding of the identified gene mutation.
- The reconstruction error from the autoencoder.

The model consists of an embedding layer for the mutation data and a few dense layers for the numerical features. The outputs are then concatenated and passed to a final output layer.
