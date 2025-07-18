import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import joblib 


df = pd.read_csv("Fire_dataset_cleaned.csv")

df['Classes'] = df['Classes'].map({'fire': 1, 'not fire': 0})

X = df.drop(columns=['Classes'])
y = df['Classes']

# Normalize numerical features using RobustScaler (more robust to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)


feature_importance = xgb_model.feature_importances_
important_features = X.columns[np.argsort(feature_importance)[-10:]]  # Top 10 features


X_train = X_train[:, np.argsort(feature_importance)[-10:]]
X_test = X_test[:, np.argsort(feature_importance)[-10:]]


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increased batch size
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(TemporalTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2
        )
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)  

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) 
        x = self.transformer(x)
        x = x.mean(dim=1)  
        x = self.dropout(x) 
        return self.fc(x)


model = TemporalTransformer(input_dim=10)  # Using top 10 features from XGBoost
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=30, patience=5):
    best_loss = float('inf')
    patience_counter = 0  # Counts epochs without improvement

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(X_batch), y_batch).item() for X_batch, y_batch in test_loader) / len(test_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  
        else:
            patience_counter += 1  

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break  # Stop training if no improvement

train_model(model, train_loader, test_loader, criterion, optimizer)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f" Model Accuracy: {acc * 100:.2f}% ")

evaluate_model(model, test_loader)


torch.save(model.state_dict(), 'temporal_transformer_model.pth')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
df = pd.read_csv("Fire_dataset_cleaned.csv")

df['Classes'] = df['Classes'].map({'fire': 1, 'not fire': 0})

X = df.drop(columns=['Classes'])
y = df['Classes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

X_train_np = np.array(X_train)

explainer_xgb = shap.Explainer(xgb_model, X_train_np)

shap_values_xgb = explainer_xgb(X_train_np)

shap.summary_plot(shap_values_xgb, X_train_np)


import torch
import numpy as np
import joblib 
from termcolor import colored 
import pandas as pd  
import torch.nn.functional as F  
model = TemporalTransformer(input_dim=10)  # Initialize the model with the same input_dim as during training
model.load_state_dict(torch.load('temporal_transformer_model.pth'))
model.eval()

scaler = joblib.load('scaler.pkl')

feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

def predict_fire_or_not():
    print("Please enter the following features for prediction:")

    user_input_dict = {}
    for feature in feature_names:
        user_input_dict[feature] = float(input(f"Enter value for {feature}: "))

    user_input_df = pd.DataFrame([user_input_dict])

    user_input_scaled = scaler.transform(user_input_df)

    user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(user_input_tensor)

        print("Raw model output (logits):", output)

        output_probabilities = F.softmax(output, dim=1)

        print("Model output probabilities:", output_probabilities)

        predicted_class = torch.argmax(output_probabilities, dim=1).item()

    print("\n" + "="*50)  # Add a separator line
    if predicted_class == 1:
        print(colored(" Prediction: Fire (1) ", 'red', attrs=['bold', 'underline']))  
    else:
        print(colored(" Prediction: No Fire (0) ", 'green', attrs=['bold', 'underline']))  
    print("="*50)  


predict_fire_or_not()
