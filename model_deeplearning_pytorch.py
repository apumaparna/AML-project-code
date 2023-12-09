import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
from google.colab import drive
import numpy as np
drive.mount('/content/drive')
%cd drive/My Drive/Colab Notebooks
# Load data
X_dev = pd.read_csv('X_dev.csv')
y_dev = pd.read_csv('y_dev.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

X = pd.concat([X_dev, X_test])
y = pd.concat([y_dev, y_test])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)  

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# DataLoader setup
train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_features, output_features)
        self.bn1 = nn.BatchNorm1d(output_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(output_features, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        # Skip connection if input and output features match
        self.skip = nn.Identity() if input_features == output_features else nn.Linear(input_features, output_features)
        self.bn_skip = nn.BatchNorm1d(output_features)

    def forward(self, x):
        identity = self.bn_skip(self.skip(x))
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class RegressionNet(nn.Module):
    def __init__(self, input_size):
        super(RegressionNet, self).__init__()
        self.res_block1 = ResidualBlock(input_size, 25)
        self.res_block2 = ResidualBlock(25, 50)
        self.fc = nn.Linear(50, 1)
        
    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.fc(x)
        return x


model = RegressionNet(X_train.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(20):  # number of epochs
    model.train()
    train_losses = []
    actuals = []
    predictions = []
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view_as(outputs))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        actuals.extend(targets.numpy())
        predictions.extend(outputs.detach().numpy()) 

    train_loss = np.mean(train_losses)
    r2_train = r2_score(actuals, predictions)

    # Validation
    model.eval()
    val_losses = []
    actuals = []
    predictions = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view_as(outputs))
            val_losses.append(loss.item())
            actuals.extend(targets.numpy())
            predictions.extend(outputs.numpy())

    val_loss = np.mean(val_losses)
    r2_val = r2_score(actuals, predictions)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train R^2: {r2_train:.4f}, Val Loss: {val_loss:.4f}, Val R^2: {r2_val:.4f}')

# Final Testing
model.eval()
test_losses = []
actuals = []
predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets.view_as(outputs))
        test_losses.append(loss.item())
        actuals.extend(targets.numpy())
        predictions.extend(outputs.numpy())

test_loss = np.mean(test_losses)
r2_test = r2_score(actuals, predictions)
print(f'Test Loss: {test_loss:.4f}, Test R^2: {r2_test:.4f}')
