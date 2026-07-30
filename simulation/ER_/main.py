import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import random

import loss_func as lfn
from utils import device


BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-3
LAMBDA      = 0.2
memory_capacity = 1000

use_replay = True
use_lars = True

device = device()

memory_x:       List[torch.Tensor] = []
memory_y:       List[torch.Tensor] = []
memory_teacher: List[torch.Tensor] = []
memory_loss:    List[float]        = []
seen_examples   = 0

data_list = ["data.csv"]
default_data_path = "MLP/DATA/"


features = ["R", "D", "H", "F"]
target = "RP"

save_model_path = "simulation/torch/my_MLP_model.pth"
save_plot_path = "simulation/torch/my_MLP_plot.png"

class MyMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, layer_num=2):
        super().__init__()
        self.input_node = nn.Linear(input_dim, hidden_dim)
        self.hidden_node = nn.Linear(hidden_dim, hidden_dim)
        self.output_node = nn.Linear(hidden_dim, output_dim)
        self.layer_num = layer_num
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_node(x)
        x = self.relu(x)
        for _ in range(self.layer_num - 1):
            x = self.hidden_node(x)
            x = self.relu(x)
        x = self.output_node(x)
        return x

model = MyMLP(input_dim=4, hidden_dim=64, output_dim=1, layer_num=2).to(device)


# try:
#     model.load_state_dict(torch.load(save_model_path))
#     train_need = False
#     if input("You have a saved model. \n" \
#     "Do you want to retrain the model? (y/n): ").lower() == 'y':
#         train_need = True
# except FileNotFoundError:
#     train_need = True
#     print("No saved model found. Training a new model.")


# # Define Data
# X = data[features].values
# y = data[target].values

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=918)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
# y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# dataset_train = TensorDataset(X_train, y_train)
# dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)


# if train_need:
#     model = MyMLP(input_dim=4, hidden_dim=64, output_dim=1)
    
#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     n_epochs = NUM_EPOCHS

#     for epoch in range(n_epochs):
#         model.train()
#         batch_losses = []
#         for batch_X, batch_y in dataloader_train:
#             optimizer.zero_grad()
#             outputs = model.forward(batch_X)
#             loss = loss_fn(outputs, batch_y)
#             batch_losses.append(loss.item())
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch+1}/{n_epochs}, Loss: {sum(batch_losses)/len(batch_losses)}")

#     torch.save(model.state_dict(), save_model_path)

# model.eval()
# with torch.no_grad():
#     val_outputs = model.forward(X_val)
#     val_loss = loss_fn(val_outputs, y_val)
#     print(f"Validation Loss: {val_loss.item()}")

# #inverse_transform the scaled features for plotting
# X_val_np = scaler.inverse_transform(X_val.numpy())
# y_val_np = y_val.numpy().flatten()
# y_pred_np = val_outputs.numpy().flatten()

# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes = axes.flatten()

# for i, feature in enumerate(features):
#     ax = axes[i]
#     ax.scatter(X_val_np[:, i], y_val_np, alpha=0.5, label='actual', color='steelblue')
#     ax.scatter(X_val_np[:, i], y_pred_np, alpha=0.5, label='predicted', color='orangered')
#     ax.set_xlabel(f'{feature}')
#     ax.set_ylabel('y')
#     ax.set_title(f'{feature} vs y')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(save_plot_path)
# plt.show()
# plt.clf()

def lars_victim() -> int:
    global memory_loss
    losses = torch.tensor(memory_loss)
    inv = 1.0 / (losses + 1e-8)
    prob = inv / inv.sum()
    return torch.multinomial(prob, 1).item()
        
def add_buffer(x, y, t, loss):
    global seen_examples, memory_x, memory_y, memory_teacher, memory_loss
    seen_examples += 1

    if len(memory_x) < memory_capacity:
        memory_x.append(x)
        memory_y.append(y)
        memory_teacher.append(t)
        memory_loss.append(loss)
        return
    
    j = random.randint(0, seen_examples - 1)
    if j < memory_capacity:
        victim = lars_victim() if use_lars else j
        memory_x[victim] = x
        memory_y[victim] = y
        memory_teacher[victim] = t
        memory_loss[victim] = loss

class tensor_dataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data[features].values, dtype=torch.float32)
        self.y = torch.tensor(data[target].values, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], 
                self.y[idx])

class dataset(Dataset):
    def __init__(self, current_ds, replay_ds=None):
        self.cur = current_ds
        self.rep = replay_ds
        self.cur_len = len(current_ds)
        self.rep_len = len(replay_ds) if replay_ds is not None else 0

    def __len__(self):
        return self.cur_len + self.rep_len

    def __getitem__(self, idx):
        if idx < self.cur_len:
            x, y = self.cur[idx]
            return x, y, torch.zeros_like(y), False
        else:
            x, y, y_t = self.rep[idx - self.cur_len]
            return x, y, y_t, True

def test_dataloader(task_ds):
    task_ds = tensor_dataset(task_ds)
    return DataLoader(
        dataset(task_ds),
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=False)

def build_dataloader(task_ds):
    replay_ds = None
    task_ds = tensor_dataset(task_ds)
    if len(memory_x) > 0:
        x_buf = torch.stack(memory_x).to(device, non_blocking=True)
        y_buf = torch.stack(memory_y).to(device, non_blocking=True)
        t_buf = torch.stack(memory_teacher).to(device, non_blocking=True)

        replay_ds = TensorDataset(x_buf, y_buf, t_buf)
        print(f"Replay buffer size: {len(replay_ds)}")
    else:
        print("Replay buffer is empty.")
    full_ds = dataset(task_ds, replay_ds)
    
    return DataLoader(
        full_ds,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=False)

def train_epoch(epoch, data_loader, optimizer, sched):
    model.train()
    total_loss = 0.0
    for X, Y, Y_t, is_rep in data_loader:
        X, Y, Y_t, is_rep = (X.to(device), 
                             Y.to(device), 
                             Y_t.to(device), 
                             is_rep.to(device))
        
        optimizer.zero_grad()
        Y_pred = model(X)

        cur_loss = lfn.MSE_loss(Y_pred, Y)
        cur_loss_per_sample = lfn.MSE_loss_per_sample(Y_pred, Y)

        if use_replay and is_rep.any():
            rep_loss = lfn.MSE_loss(Y_t, Y)
            total_loss_batch = LAMBDA * cur_loss + (1 - LAMBDA) * rep_loss
        else:
            total_loss_batch = cur_loss
        
        teacher = Y_pred.detach().clone()
        total_loss_batch.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for i in range(X.size(0)):
            add_buffer(
                X[i].detach().cpu(), 
                Y[i].detach().cpu(), 
                teacher[i].detach().cpu(),
                cur_loss_per_sample[i].item()
            )
        total_loss += total_loss_batch.item()
    sched.step()
    return total_loss / len(data_loader)


def train_on_task(train_ds, task_name):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    data = build_dataloader(train_ds)
    for ep in range(1, NUM_EPOCHS + 1):
        print(f"Training on task: {task_name}, Epoch: {ep}/{NUM_EPOCHS}")
        ep_loss = train_epoch(ep, data, optimizer, sched)
        print(f"Task: {task_name} | Epoch: {ep}/{NUM_EPOCHS} | Loss: {ep_loss:.4f}")



def data_split(csv_data, val_size=0.2):
    data = pd.read_csv(default_data_path + csv_data)
    train, val = train_test_split(data, test_size=val_size, random_state=918)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    return train, val


if __name__ == "__main__":
    val_datasets = {}
    for data_file in data_list:
        train_data, val_data = data_split(data_file)
        val_datasets[data_file] = val_data
        train_on_task(train_data, task_name=data_file)
    
    for data_file, val_data in val_datasets.items():
        val_loader = test_dataloader(val_data)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X, Y, _, _ in val_loader:
                X, Y = X.to(device), Y.to(device)
                Y_pred = model(X)
                loss = lfn.MSE_loss(Y_pred, Y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss for {data_file}: {avg_val_loss:.4f}")
