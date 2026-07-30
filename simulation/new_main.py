import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

import pandas as pd
from typing import List, Tuple
import random
import datetime

import ER_.loss_func as lfn
from ER_.utils import device as get_device
from csv_factory import create_CSV
from environments.transmitter import Transmitter, all_transmitters, new_transmitters


# ── 하이퍼파라미터 ────────────────────────────────────────────────
BATCH_SIZE       = 32
NUM_EPOCHS       = 30
LR               = 1e-3
LAMBDA           = 1.0
memory_capacity  = 10000

use_replay = True
use_lars   = False

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

# ── 장치 설정 ─────────────────────────────────────────────────────
device = get_device()

# ── 리플레이 버퍼 ─────────────────────────────────────────────────
memory_x:       List[torch.Tensor] = []
memory_y:       List[torch.Tensor] = []
memory_teacher: List[torch.Tensor] = []
memory_loss:    List[float]        = []
seen_examples = 0

# ── 데이터 설정 ───────────────────────────────────────────────────
default_data_path = "MLP/DATA/"
features          = ["R", "D", "H", "F"]
target            = "RP"

# ── CSV 생성 설정 ─────────────────────────────────────────────────
# (transmitter 그룹, 주파수 리스트, 저장 파일명) 형태로 태스크 정의
# 태스크를 추가하려면 아래 리스트에 항목을 추가하면 됩니다.
empt_trans = ""
empt_freq = []
task3_trans = [
    Transmitter("MBC-ChungJu", 127.433977355451, 36.61907632039761),
    Transmitter('Broad-ChungJu', 127.47905459727248, 36.63418678335862)
]

task_configs: List[Tuple] = [
    (empt_trans, empt_freq, "data.csv"),
    (all_transmitters, [100_000_000], "task1.csv"),
    (new_transmitters, [50_000_000], "task2.csv"),
    (task3_trans, [200_000_000], "task3.csv")
    # (ex_transmitters, [200_000_000], "task_ex.csv"),  # 추가 예시
]

# ── 저장 경로 ─────────────────────────────────────────────────────
save_model_dir  = "simulation/ER_/model/"
save_model_path = os.path.join(save_model_dir, "model.pth")
os.makedirs(save_model_dir, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# 모델 정의
# ════════════════════════════════════════════════════════════════════
class MyMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, layer_num=2):
        super().__init__()
        self.input_node  = nn.Linear(input_dim, hidden_dim)
        self.hidden_node = nn.Linear(hidden_dim, hidden_dim)
        self.output_node = nn.Linear(hidden_dim, output_dim)
        self.layer_num   = layer_num
        self.relu        = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_node(x))
        for _ in range(self.layer_num - 1):
            x = self.relu(self.hidden_node(x))
        return self.output_node(x)


model = MyMLP(input_dim=4, hidden_dim=64, output_dim=1, layer_num=2).to(device)


# ════════════════════════════════════════════════════════════════════
# 모델 저장 / 불러오기
# ════════════════════════════════════════════════════════════════════
def save_model(model, path, epoch=None, optimizer=None, loss=None):
    """
        - model_state_dict : 모델 파라미터
        - epoch            : 마지막 학습 epoch (선택)
        - optimizer_state  : optimizer 상태 (선택, 이어서 학습 시 필요)
        - loss             : 마지막 loss 값 (선택)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch':            epoch,
        'optimizer_state':  optimizer.state_dict() if optimizer else None,
        'loss':             loss,
    }
    torch.save(checkpoint, path)
    print(f"[저장] 모델 저장 완료 → {path}")


def load_model(model, path, optimizer=None):
    """
    Returns:
        start_epoch (int): 이어서 학습할 epoch 번호
        loss        (float): 마지막 저장 시 loss
    """
    if not os.path.exists(path):
        print(f"[불러오기] 저장된 모델 없음 → 새로 학습합니다.")
        return 0, None

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and checkpoint.get('optimizer_state'):
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    epoch = checkpoint.get('epoch', 0)
    loss  = checkpoint.get('loss', None)
    print(f"[불러오기] 모델 불러오기 완료 ← {path}  (epoch={epoch}, loss={loss})")
    return epoch, loss


# ════════════════════════════════════════════════════════════════════
# 버퍼 관련 함수
# ════════════════════════════════════════════════════════════════════
def lars_victim() -> int:
    losses = torch.tensor(memory_loss, dtype=torch.float)
    inv    = 1.0 / (losses + 1e-8)
    prob   = inv / inv.sum()
    return torch.multinomial(prob, 1).item()


def add_buffer(x, y, t, loss: float):
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
        memory_x[victim]       = x
        memory_y[victim]       = y
        memory_teacher[victim] = t
        memory_loss[victim]    = loss


# ════════════════════════════════════════════════════════════════════
# Dataset 정의
# ════════════════════════════════════════════════════════════════════
class tensor_dataset(Dataset):
    """DataFrame → Tensor Dataset 변환"""
    def __init__(self, data: pd.DataFrame):
        self.X = torch.tensor(data[features].values, dtype=torch.float32)
        self.y = torch.tensor(data[target].values,   dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class replay_dataset(Dataset):
    """current_ds + replay_ds 혼합 Dataset"""
    def __init__(self, current_ds: Dataset, replay_ds: Dataset = None):
        self.cur     = current_ds
        self.rep     = replay_ds
        self.cur_len = len(current_ds)
        self.rep_len = len(replay_ds) if replay_ds is not None else 0

    def __len__(self):
        return self.cur_len + self.rep_len

    def __getitem__(self, idx):
        if idx < self.cur_len:
            x, y = self.cur[idx]
            return x, y, torch.zeros_like(y), False   # is_replay=False
        else:
            x, y, y_t = self.rep[idx - self.cur_len]
            return x, y, y_t, True                    # is_replay=True


# ════════════════════════════════════════════════════════════════════
# DataLoader 빌더
# ════════════════════════════════════════════════════════════════════
def build_dataloader(task_ds: pd.DataFrame) -> DataLoader:
    """학습용: current + replay 혼합 DataLoader"""
    cur_ds     = tensor_dataset(task_ds)
    replay_ds  = None

    if len(memory_x) > 0:
        x_buf = torch.stack(memory_x).to(device, non_blocking=True)
        y_buf = torch.stack(memory_y).to(device, non_blocking=True)
        t_buf = torch.stack(memory_teacher).to(device, non_blocking=True)
        replay_ds = TensorDataset(x_buf, y_buf, t_buf)
        print(f"  Replay buffer size: {len(replay_ds)}")
    else:
        print("  Replay buffer is empty.")

    full_ds = replay_dataset(cur_ds, replay_ds)
    return DataLoader(full_ds, batch_size=BATCH_SIZE,
                      shuffle=True, pin_memory=False)


def val_dataloader(task_ds: pd.DataFrame) -> DataLoader:
    """평가용: current 데이터만 DataLoader"""
    cur_ds = tensor_dataset(task_ds)
    full_ds = replay_dataset(cur_ds)          # replay 없음
    return DataLoader(full_ds, batch_size=BATCH_SIZE,
                      shuffle=False, pin_memory=False)


# ════════════════════════════════════════════════════════════════════
# 학습 함수
# ════════════════════════════════════════════════════════════════════
def train_epoch(data_loader, optimizer) -> float:
    model.train()
    total_loss = 0.0

    for X, Y, Y_t, is_rep in data_loader:
        X, Y, Y_t, is_rep = (X.to(device), Y.to(device),
                              Y_t.to(device), is_rep.to(device))
        optimizer.zero_grad()
        Y_pred = model(X)

        cur_mask = ~is_rep
        cur_loss_per_sample = lfn.MSE_loss_per_sample(Y_pred, Y)  # (B,)

        if use_replay and is_rep.any() and cur_mask.any():
            # current 샘플 loss
            cur_loss = lfn.MSE_loss(Y_pred[cur_mask], Y[cur_mask])
            # replay 샘플 loss: 현재 모델 예측 vs 정답
            rep_loss = lfn.MSE_loss(Y_pred[is_rep],  Y[is_rep])
            total_loss_batch = cur_loss + LAMBDA * rep_loss
        else:
            total_loss_batch = lfn.MSE_loss(Y_pred, Y)

        teacher = Y_pred.detach().clone()

        total_loss_batch.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for i in range(X.size(0)):
            add_buffer(
                X[i].detach().cpu(),
                Y[i].detach().cpu(),
                teacher[i].detach().cpu(),
                cur_loss_per_sample[i].item()   # float로 변환
            )
        total_loss += total_loss_batch.item()

    return total_loss / len(data_loader)


def train_on_task(train_ds: pd.DataFrame, task_name: str):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    sched     = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    print(f"\n=== Task: {task_name} ===")
    data_loader = build_dataloader(train_ds)

    best_loss = float('inf')

    for ep in range(1, NUM_EPOCHS + 1):
        ep_loss = train_epoch(data_loader, optimizer)
        sched.step()                              # epoch 단위 lr 갱신
        print(f"  Epoch {ep:>2}/{NUM_EPOCHS} | Loss: {ep_loss:.4f}")

        # ── best loss일 때 모델 저장 ──────────────────────────
        if ep_loss < best_loss:
            best_loss = ep_loss
            save_model(model, save_model_path,
                       epoch=ep, optimizer=optimizer, loss=best_loss)


# ════════════════════════════════════════════════════════════════════
# 평가 함수
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(task_name: str, val_ds: pd.DataFrame):
    loader = val_dataloader(val_ds)
    model.eval()

    total_mse  = 0.0
    total_mae  = 0.0
    n_batches  = len(loader)

    for X, Y, _, _ in loader:
        X, Y   = X.to(device), Y.to(device)
        Y_pred = model(X)
        total_mse += lfn.MSE_loss(Y_pred, Y).item()
        total_mae += (Y_pred - Y).abs().mean().item()

    avg_mse  = total_mse / n_batches
    avg_rmse = avg_mse ** 0.5
    avg_mae  = total_mae / n_batches

    print(f"\n[평가] {task_name}")
    print(f"  MSE  : {avg_mse:.4f}")
    print(f"  RMSE : {avg_rmse:.4f} dBm")
    print(f"  MAE  : {avg_mae:.4f} dBm")

    return {'mse': avg_mse, 'rmse': avg_rmse, 'mae': avg_mae}


# ════════════════════════════════════════════════════════════════════
# 데이터 분리
# ════════════════════════════════════════════════════════════════════
def data_split(df: pd.DataFrame, val_size: float = 0.2):
    """
    DataFrame을 train/val로 분리합니다.
    csv_factory가 이미 shuffle된 DataFrame을 반환하므로
    read_csv 없이 DataFrame을 직접 받습니다.
    """
    train, val = train_test_split(df, test_size=val_size, random_state=918)
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    return train, val


# ════════════════════════════════════════════════════════════════════
# CSV 생성 및 파일명 등록
# ════════════════════════════════════════════════════════════════════
def generate_csv_tasks(task_configs: List[Tuple]) -> List[Tuple[str, pd.DataFrame]]:
    """
    task_configs의 각 항목에 대해 CSV를 생성하고
    (파일명, DataFrame) 리스트를 반환합니다.

    - 동일한 파일명의 CSV가 이미 존재하면 생성하지 않고 로드합니다.
    - 존재하지 않으면 create_CSV로 새로 생성 후 저장합니다.

    task_configs 형식:
        [(transmitters, frequency_list, file_name), ...]
    """
    os.makedirs(default_data_path, exist_ok=True)
    tasks = []

    for transmitters, frequency_list, file_name in task_configs:
        save_path = os.path.join(default_data_path, file_name)

        if os.path.exists(save_path):
            # ── 파일이 이미 존재하면 로드 ─────────────────────
            print(f"\n[CSV 로드] {file_name} 이미 존재 → 로드합니다.")
            df = pd.read_csv(save_path, index_col=0)
            df = df.reset_index(drop=True)
            print(f"[CSV 로드] {file_name} 완료 ({len(df)}행)")
        else:
            # ── 파일이 없으면 새로 생성 ───────────────────────
            print(f"\n[CSV 생성] {file_name} 생성 중...")
            df = create_CSV(
                transmitters   = transmitters,
                frequency_list = frequency_list,
                save_path      = save_path,
            )
            print(f"[CSV 생성] {file_name} 완료 → {save_path} ({len(df)}행)")

        tasks.append((file_name, df))

    return tasks


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. CSV 생성 ───────────────────────────────────────────────
    print("=== CSV 데이터 생성 ===")
    tasks = generate_csv_tasks(task_configs)
    # tasks: [("task1.csv", df1), ("task2.csv", df2), ...]

    # ── 2. train/val 분리 및 저장 ────────────────────────────────
    val_datasets = {}   # { file_name: val_df }
    for file_name, df in tasks:
        train_data, val_data = data_split(df)
        val_datasets[file_name] = val_data
        print(f"  {file_name}: train={len(train_data)}, val={len(val_data)}")

    # ── 3. 태스크 순서대로 ER 학습 ───────────────────────────────
    print("\n=== Experience Replay 학습 시작 ===")
    for file_name, df in tasks:
        train_data, _ = data_split(df)
        train_on_task(train_data, task_name=file_name)

    # ── 4. 최종 모델 저장 ─────────────────────────────────────────
    final_path = os.path.join(save_model_dir, f"model_{date}.pth")
    save_model(model, final_path, epoch=NUM_EPOCHS, loss=None)

    # ── 5. 최종 평가 ──────────────────────────────────────────────
    print("\n=== 최종 평가 ===")
    results = {}
    for file_name, val_data in val_datasets.items():
        results[file_name] = evaluate(file_name, val_data)

    # ── 6. 결과 CSV 저장 ──────────────────────────────────────────
    csv_path = os.path.join(save_model_dir, f"results_{date}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'MSE', 'RMSE(dBm)', 'MAE(dBm)'])
        for task, res in results.items():
            writer.writerow([task,
                             f"{res['mse']:.4f}",
                             f"{res['rmse']:.4f}",
                             f"{res['mae']:.4f}"])
    print(f"\n결과 저장 완료 → {csv_path}")