# AI-Based Wildfire Prediction Using Climate Data

This project implements a robust wildfire detection system using a hybrid approach combining traditional machine learning (XGBoost) for feature selection and a deep learning-based Temporal Transformer model for classification. It includes SHAP-based interpretability and a command-line interface (CLI) for real-time wildfire prediction.

## Project Structure

wildfire-prediction-transformer/

├── final_wf.py # Main training, evaluation, and prediction script

├── Fire_dataset_cleaned.csv # Cleaned input dataset

├── temporal_transformer_model.pth # Trained transformer model weights

├── scaler.pkl # Scaler for data normalization

├── README.md # Project documentation

markdown
Copy code

## Features

- XGBoost-based feature importance analysis
- Transformer-based sequence classification model (PyTorch)
- SHAP interpretability for model explanations
- RobustScaler for outlier-resistant preprocessing
- CLI interface for wildfire prediction

## Dataset

The dataset (`Fire_dataset_cleaned.csv`) includes the following features:
- Temperature
- RH (Relative Humidity)
- Ws (Wind speed)
- Rain
- Fire indices: FFMC, DMC, DC, ISI, BUI, FWI
- Classes (Label: Fire or No Fire)

## Model Architecture

### XGBoost
Used to identify the top 10 most important features through feature importance ranking.

### Temporal Transformer
A PyTorch-based Transformer encoder model for binary classification:
- Includes dropout regularization
- Applies early stopping for improved generalization
- Uses positional encoding and attention mechanisms

## Training and Evaluation

To train and evaluate the model, run:

```bash
python final_wf.py
This script performs:

Data preprocessing

Feature selection using XGBoost

Transformer model training and evaluation

Accuracy and metrics reporting

Explainability
SHAP (SHapley Additive exPlanations) is used to interpret predictions made by the XGBoost model. It provides:

Summary plots of feature contributions

Visual insights into how features affect predictions

CLI Prediction Interface
After training, the same script (final_wf.py) can be used to run an interactive command-line interface for predictions:

bash
Copy code
python final_wf.py
You will be prompted to input values for selected features. The model will return:

Raw logits

Softmax probabilities

Final prediction: Fire / No Fire

Saved Files
temporal_transformer_model.pth: Trained model weights

scaler.pkl: Fitted scaler used for input normalization

Requirements
Install required dependencies using:

bash
Copy code
pip install -r requirements.txt
Dependencies:
torch

xgboost

pandas

numpy

scikit-learn

shap

termcolor

joblib

Notes
Feature selection improves model efficiency and interpretability

Model uses early stopping and dropout to prevent overfitting

CLI makes prediction easy without needing a GUI

Future Enhancements
Web-based deployment using Flask or Streamlit

Real-time predictions using weather API data

Alerting system for high-risk zones

Author
Eluru Poojith Kumar Reddy
