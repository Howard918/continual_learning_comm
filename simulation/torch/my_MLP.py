import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("MLP/DATA/data.csv")

features = ["R", "D", "H", "F"]
target = "RP"

save_model_path = "simulation/torch/my_MLP_model.pth"
save_plot_path = "simulation/torch/my_MLP_plot.png"

class MyMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super().__init__()
        self.input_node = nn.Linear(input_dim, hidden_dim)
        self.hidden_node = nn.Linear(hidden_dim, hidden_dim)
        self.output_node = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_node(x)
        x = self.relu(x)
        x = self.hidden_node(x)
        x = self.relu(x)
        x = self.output_node(x)
        return x

try:
    model = MyMLP(input_dim=4, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(save_model_path))
    train_need = False
    if input("You have a saved model. \n" \
    "Do you want to retrain the model? (y/n): ").lower() == 'y':
        train_need = True
except FileNotFoundError:
    train_need = True
    print("No saved model found. Training a new model.")


# Define Data
X = data[features].values
y = data[target].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=918)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

dataset_train = TensorDataset(X_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

criterion = nn.MSELoss()

if train_need:
    model = MyMLP(input_dim=4, hidden_dim=64, output_dim=1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 200

    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        for batch_X, batch_y in dataloader_train:
            optimizer.zero_grad()
            outputs = model.forward(batch_X)
            loss = criterion(outputs, batch_y)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {sum(batch_losses)/len(batch_losses)}")

    torch.save(model.state_dict(), save_model_path)

model.eval()
with torch.no_grad():
    val_outputs = model.forward(X_val)
    val_loss = criterion(val_outputs, y_val)
    print(f"Validation Loss: {val_loss.item()}")

#inverse_transform the scaled features for plotting
X_val_np = scaler.inverse_transform(X_val.numpy())
y_val_np = y_val.numpy().flatten()
y_pred_np = val_outputs.numpy().flatten()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]
    ax.scatter(X_val_np[:, i], y_val_np, alpha=0.5, label='actual', color='steelblue')
    ax.scatter(X_val_np[:, i], y_pred_np, alpha=0.5, label='predicted', color='orangered')
    ax.set_xlabel(f'{feature}')
    ax.set_ylabel('y')
    ax.set_title(f'{feature} vs y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_plot_path)
plt.show()
plt.clf()