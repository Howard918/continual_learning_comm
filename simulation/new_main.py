import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import MinMaxScaler
import joblib

import pandas as pd
from typing import List, Tuple, Dict
import random
import datetime

import matplotlib
matplotlib.use("Agg")   # 화면 없는 환경(서버)에서도 저장 가능하도록
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

import ER_.loss_func as lfn
from ER_.utils import device as get_device
from csv_factory import create_CSV
from environments.transmitter import Transmitter, all_transmitters, new_transmitters


# ── 하이퍼파라미터 ────────────────────────────────────────────────
BATCH_SIZE       = 256
NUM_EPOCHS       = 200
LR               = 1e-3
LAMBDA           = 1.0
memory_capacity  = 5000
VAL_SIZE         = 0.2     # train 내부에서 val로 떼어낼 비율
TEST_SIZE        = 0.2     # 전체 데이터에서 test로 떼어낼 비율

use_replay = False
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

# ── 태스크별 epoch 학습 이력 (train/val loss) 저장 ──────────────────
#     history_all[task_name] = {"train": [...], "val": [...]}
history_all: Dict[str, Dict[str, List[float]]] = {}

# ── 데이터 설정 ───────────────────────────────────────────────────
default_data_path = "MLP/DATA/"
features          = ["R", "D", "H", "F"]
target            = "PL"

# ── CSV 생성 설정 ─────────────────────────────────────────────────
# (transmitter 그룹, 주파수 리스트, 저장 파일명) 형태로 태스크 정의
# 태스크를 추가하려면 아래 리스트에 항목을 추가하면 됩니다.
empt_trans = ""
freq = [100_000_000]
CheongJu = [
    Transmitter("MBC-CheongJu", 127.433977355451, 36.61907632039761),
    Transmitter('Broad-CheongJu', 127.47905459727248, 36.63418678335862)
]
DaeJeon = [
    Transmitter("MBC-DaeJeon", 127.397198121239, 36.3760857047283),
    Transmitter("KBS-DaeJeon", 127.380567392303, 36.3704437169546),
    Transmitter("CMB-DaeJeon", 127.419676653034, 36.3341326962576)
]
ChungJu = [
    Transmitter("MBC-ChungJu", 127.924378514041, 36.9585291745712),
    Transmitter("KBS-ChungJu", 127.920483843397, 36.9724980330778)
]
Seoul = [
    Transmitter("KBS_main", 126.916716838156, 37.5259698897016),
    Transmitter("SBS", 126.87374657727, 37.5291902429029),
    Transmitter("MBC_sa", 126.890988995582, 37.5811234199086),
]
GangNeung = [
    Transmitter("KBS-KangNeung", 128.891256884067, 37.7520385140085),
    Transmitter("MBC-KangNeung", 128.904230376246, 37.7709174571674)
]
DaeGu = [
    Transmitter("TBN-DaeGu", 128.580363461352, 35.843420581465),
    Transmitter("TBC-DaeGu", 128.622440960025, 35.8323075963103)
]
JeonJu = [
    Transmitter("JeonJu Radio", 127.158316016663, 35.8489105505397),
    Transmitter("KBS-JeonJu", 127.104790219087, 35.8221763698401)
]
JeonNam_GwangJu = [
    Transmitter("KBS-MokPo", 126.394348103215, 34.8120944953763),
    Transmitter("KBS-SunCheon", 127.48480049784, 34.9683819549619),
    Transmitter("KBS-GwangJu", 126.854624825405, 35.1581118750609)
]


task_configs: List[Tuple] = [
    (Seoul, freq, "Seoul.csv"),
    (GangNeung, freq, "GanNeung.csv"),
    (DaeJeon, freq, "DaeJun.csv"),
    (CheongJu, freq, "CheongJu.csv"),
    (ChungJu, freq, "ChungJu.csv"),
    (DaeGu, freq, "DaeGu.csv"),
    (JeonJu, freq, "JeonJu.csv"),
    (JeonNam_GwangJu, freq, "JeonNam_GwangJu.csv")
]

# ── 저장 경로 ─────────────────────────────────────────────────────
save_model_dir  = "simulation/ER_/model/"
plot_dir        = os.path.join(save_model_dir, "plots")
history_dir     = os.path.join(save_model_dir, "history")
save_model_path = os.path.join(save_model_dir, "model.pth")
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# 한글 폰트 설정 (그래프 라벨 깨짐 방지)
# ════════════════════════════════════════════════════════════════════
def _setup_korean_font() -> bool:
    system = platform.system()
    candidates = {
        "Windows": ["Malgun Gothic"],
        "Darwin":  ["AppleGothic"],
        "Linux":   ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"],
    }.get(system, [])
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


USE_KOREAN_LABELS = _setup_korean_font()


def L(korean: str, english: str) -> str:
    """한글 폰트가 없으면 자동으로 영문 라벨을 사용합니다."""
    return korean if USE_KOREAN_LABELS else english


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
        - loss             : 마지막 loss 값 (선택, val_loss 기준)
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
    """학습용: current(train) + replay 혼합 DataLoader"""
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
    """
    검증/평가용: replay 없이 순수 데이터만 담은 DataLoader.
    - train_on_task 내부에서 매 epoch validation loss 계산용으로 사용
    - 최종 test 평가(evaluate)에서도 재사용
    """
    cur_ds  = tensor_dataset(task_ds)
    full_ds = replay_dataset(cur_ds)          # replay 없음
    return DataLoader(full_ds, batch_size=BATCH_SIZE,
                      shuffle=False, pin_memory=False)


# ════════════════════════════════════════════════════════════════════
# 학습 / 검증 함수
# ════════════════════════════════════════════════════════════════════
def train_epoch(data_loader, optimizer) -> float:
    """한 epoch 동안 (current + replay) 배치로 학습하고 평균 train loss 반환."""
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


@torch.no_grad()
def compute_avg_loss(loader: DataLoader) -> float:
    """
    replay 없이 순수 MSE loss만 계산합니다.
    (매 epoch validation loss / 최종 test loss 계산에 공통 사용)
    """
    model.eval()
    total_loss = 0.0
    n_batches  = len(loader)

    for X, Y, _, _ in loader:
        X, Y   = X.to(device), Y.to(device)
        Y_pred = model(X)
        total_loss += lfn.MSE_loss(Y_pred, Y).item()

    return total_loss / max(n_batches, 1)


# ════════════════════════════════════════════════════════════════════
# 태스크별 loss curve 시각화
# ════════════════════════════════════════════════════════════════════
def plot_task_history(task_name: str, train_losses: List[float], val_losses: List[float]):
    """
    한 태스크의 epoch별 train loss / val loss 를 하나의 그래프로 저장합니다.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label=L("Train Loss", "Train Loss"),
             color="steelblue", marker='o', markersize=3)
    plt.plot(epochs, val_losses, label=L("Val Loss", "Val Loss"),
             color="orangered", marker='x', markersize=3)
    plt.xlabel(L("Epoch", "Epoch"))
    plt.ylabel(L("MSE Loss", "MSE Loss"))
    plt.title(L(f"[{task_name}] Epoch별 Train / Val Loss",
                f"[{task_name}] Train / Val Loss per Epoch"))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = task_name.replace(".csv", "")
    path = os.path.join(plot_dir, f"{safe_name}_loss_curve_{date}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[그래프] {task_name} train/val loss curve 저장 → {path}")

    # epoch별 수치도 CSV로 함께 저장 (재분석 용도)
    hist_df = pd.DataFrame({
        "epoch":      list(epochs),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    })
    hist_csv = os.path.join(history_dir, f"{safe_name}_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"[기록] {task_name} epoch별 loss 기록 저장 → {hist_csv}")


def plot_all_tasks_history(history_all: Dict[str, Dict[str, List[float]]]):
    """
    모든 태스크의 train/val loss curve 를 한 화면에 서브플롯으로 모아
    전체 흐름을 한 번에 비교할 수 있도록 저장합니다.
    """
    task_names = list(history_all.keys())
    n = len(task_names)
    if n == 0:
        return

    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, task_name in enumerate(task_names):
        ax = axes[idx]
        train_losses = history_all[task_name]["train"]
        val_losses   = history_all[task_name]["val"]
        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, label="Train", color="steelblue")
        ax.plot(epochs, val_losses,   label="Val",   color="orangered")
        ax.set_title(task_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 남는 subplot 비우기
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    path = os.path.join(plot_dir, f"all_tasks_loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[그래프] 전체 태스크 train/val loss curve 저장 → {path}")


# ════════════════════════════════════════════════════════════════════
# [신규] 통합 test set 기준 continual-learning 성능 곡선 시각화
# ════════════════════════════════════════════════════════════════════
def plot_continual_test_curve(records: List[Dict]):
    """
    records: [{"stage": 0, "stage_task": "Seoul.csv", "mse":.., "rmse":.., "mae":..}, ...]

    태스크를 하나씩 학습해 나가면서, 매번 "동일한 통합 test set" 에
    대해 측정한 MSE/RMSE 의 변화를 하나의 곡선으로 그립니다.
    ER이 과거 지식을 잘 보존하고 있다면 이 곡선은 태스크가 늘어나도
    급격히 나빠지지 않고 완만하게 유지/개선되는 형태를 보여야 합니다.
    """
    if not records:
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(df["stage"], df["mse"], marker='o', color="steelblue")
    axes[0].set_xticks(df["stage"])
    axes[0].set_xticklabels(df["stage_task"], rotation=30, ha='right')
    axes[0].set_ylabel("MSE")
    axes[0].set_title(L("통합 Test Set 기준 MSE 변화",
                        "MSE on Unified Test Set over Tasks"))
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["stage"], df["rmse"], marker='o', color="orangered")
    axes[1].set_xticks(df["stage"])
    axes[1].set_xticklabels(df["stage_task"], rotation=30, ha='right')
    axes[1].set_ylabel("RMSE (dBm)")
    axes[1].set_title(L("통합 Test Set 기준 RMSE 변화",
                        "RMSE on Unified Test Set over Tasks"))
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, f"continual_test_curve_{date}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[그래프] 통합 test set 기준 continual-learning 곡선 저장 → {path}")

    csv_path = os.path.join(history_dir, f"continual_test_curve_{date}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[기록] 통합 test set 평가 기록 저장 → {csv_path}")


# ════════════════════════════════════════════════════════════════════
# 태스크 학습 (train → train/val 분리, val로 매 epoch 검증)
# ════════════════════════════════════════════════════════════════════
def train_on_task(train_ds: pd.DataFrame, task_name: str, val_size: float = VAL_SIZE):
    """
    train_ds (해당 태스크의 학습용 데이터, 이미 "이 태스크 자신의"
    MinMaxScaler로 정규화된 상태)를 다시 train/val로 분리합니다.
        - train : 실제 파라미터 업데이트 + 리플레이 버퍼 채우기에 사용
        - val   : 매 epoch 학습 이후 성능 확인(검증)에만 사용, 학습에는 관여하지 않음
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    sched     = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── train 내부에서 train/val 분리 ───────────────────────────────
    inner_train, inner_val = train_test_split(
        train_ds, test_size=val_size, shuffle=True, random_state=918
    )
    inner_train = inner_train.reset_index(drop=True)
    inner_val   = inner_val.reset_index(drop=True)

    print(f"\n=== Task: {task_name} ===")
    print(f"  inner_train={len(inner_train)}, inner_val={len(inner_val)}")

    # 학습용 로더: (inner_train + replay buffer) 혼합
    train_loader = build_dataloader(inner_train)
    # 검증용 로더: inner_val만 (replay 없음, 학습에 관여하지 않음)
    val_loader   = val_dataloader(inner_val)

    best_val_loss = float('inf')
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for ep in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(train_loader, optimizer)
        sched.step()                                   # epoch 단위 lr 갱신

        val_loss = compute_avg_loss(val_loader)         # ← 매 epoch validation

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {ep:>3}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} "
             f"| Val Loss: {val_loss:.4f}")

        # ── best val_loss 기준으로 모델 저장 ───────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_model_path,
                      epoch=ep, optimizer=optimizer, loss=best_val_loss)

    load_model(model, save_model_path)

    # ── 태스크별 이력 저장 및 시각화 ────────────────────────────────
    history_all[task_name] = {"train": train_losses, "val": val_losses}
    plot_task_history(task_name, train_losses, val_losses)


# ════════════════════════════════════════════════════════════════════
# 평가 함수 (통합 test set에 대해, 매 태스크 학습 직후 반복 호출됨)
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(task_name: str, test_ds: pd.DataFrame):
    """
    현재 시점의 model을, 넘겨받은 test_ds(항상 동일한 통합 test set)로
    평가합니다. ER 학습이 진행되며 태스크가 하나씩 추가될 때마다
    동일한 벤치마크로 반복 호출되어, 과거 지식이 얼마나 보존되는지
    (catastrophic forgetting 여부)를 추적하는 용도로 사용됩니다.
    """
    loader = val_dataloader(test_ds)
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

    print(f"\n[통합 Test 평가] {task_name}")
    print(f"  MSE  : {avg_mse:.4f}")
    print(f"  RMSE : {avg_rmse:.4f} dBm")
    print(f"  MAE  : {avg_mae:.4f} dBm")

    return {'mse': avg_mse, 'rmse': avg_rmse, 'mae': avg_mae}


# ════════════════════════════════════════════════════════════════════
# 데이터 분리 (전체 데이터 → train 전체 / test)
# ════════════════════════════════════════════════════════════════════
def data_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    DataFrame을 train(전체) / test 로 분리합니다. (아직 스케일링 전 원본 상태)

    - train : train_on_task() 내부에서 다시 train/val로 나뉘어 사용됩니다.
    - test  : 다른 모든 태스크의 test와 하나로 합쳐져 "통합 test set"을
              구성하며, 학습/검증 과정에는 절대 사용되지 않습니다.
    """
    train, test = train_test_split(df, test_size=test_size, shuffle=True, random_state=918)
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    return train, test


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
    # tasks: [("Seoul.csv", df1), ("GanNeung.csv", df2), ...]

    # ── 2. 각 태스크를 train(전체) / test로 우선 분리 (스케일링 전) ──
    raw_train: Dict[str, pd.DataFrame] = {}
    raw_test:  Dict[str, pd.DataFrame] = {}
    for file_name, df in tasks:
        train_data, test_data = data_split(df)
        raw_train[file_name] = train_data
        raw_test[file_name]  = test_data
        print(f"  {file_name}: train(total)={len(train_data)}, test={len(test_data)}")

    # ── 3. [수정] 태스크(데이터셋)별로 각각 독립된 스케일러를 fit ──
    scalers: Dict[str, MinMaxScaler] = {}
    train_datasets: Dict[str, pd.DataFrame] = {}
    scaled_test_list: List[pd.DataFrame] = []

    scaler_dir = os.path.join(save_model_dir, "scalers")
    os.makedirs(scaler_dir, exist_ok=True)

    for file_name in raw_train.keys():
        train_data = raw_train[file_name].copy()
        test_data  = raw_test[file_name].copy()

        task_scaler = MinMaxScaler()
        task_scaler.fit(train_data[features])

        train_data[features] = task_scaler.transform(train_data[features])
        test_data[features]  = task_scaler.transform(test_data[features])

        scalers[file_name]        = task_scaler
        train_datasets[file_name] = train_data
        scaled_test_list.append(test_data)

        # 나중에 추론/역변환(inverse_transform)에 쓸 수 있도록 저장
        joblib.dump(task_scaler,
                   os.path.join(scaler_dir, f"{file_name.replace('.csv','')}_scaler.pkl"))
        print(f"  [스케일러] {file_name} 전용 MinMaxScaler fit 완료 "
             f"(train {len(train_data)}행 기준)")

    # ── 4. 모든 태스크의 test를 하나로 합쳐 "통합 test set" 구성 ──
    combined_test_df = pd.concat(scaled_test_list, ignore_index=True)
    print(f"[통합 Test Set] 전체 태스크 test를 합쳐 총 {len(combined_test_df)}행 구성")

    # ── 5. 태스크 순서대로 ER 학습 + 매 태스크 직후 통합 test set 평가 ──
    print("\n=== Experience Replay 학습 시작 ===")
    continual_eval_records: List[Dict] = []

    for stage, (file_name, train_data) in enumerate(train_datasets.items()):
        train_on_task(train_data, task_name=file_name, val_size=VAL_SIZE)

        stage_result = evaluate(f"stage{stage}_{file_name}", combined_test_df)
        continual_eval_records.append({
            "stage":      stage,
            "stage_task": file_name,
            "mse":        stage_result["mse"],
            "rmse":       stage_result["rmse"],
            "mae":        stage_result["mae"],
        })

    # ── 6. 전체 태스크 train/val loss curve 종합 시각화 ──────────
    plot_all_tasks_history(history_all)

    # ── 7. 통합 test set 기준 continual-learning 성능 곡선 시각화 ──
    plot_continual_test_curve(continual_eval_records)

    # ── 8. 최종 모델 저장 ─────────────────────────────────────────
    final_path = os.path.join(save_model_dir, f"model_{date}.pth")
    save_model(model, final_path, epoch=NUM_EPOCHS, loss=None)

    # ── 9. 최종 결과 CSV 저장 (마지막 stage 결과 = 전체 학습 완료 후 결과) ──
    final_result = continual_eval_records[-1]
    csv_path = os.path.join(save_model_dir, f"results_{date}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Task', 'MSE', 'RMSE(dBm)', 'MAE(dBm)'])
        for rec in continual_eval_records:
            writer.writerow([rec['stage'], rec['stage_task'],
                             f"{rec['mse']:.4f}",
                             f"{rec['rmse']:.4f}",
                             f"{rec['mae']:.4f}"])
    print(f"\n결과 저장 완료 → {csv_path}")
    print(f"최종(전체 학습 완료 후) 통합 test set 성능: "
         f"MSE={final_result['mse']:.4f} | "
         f"RMSE={final_result['rmse']:.4f} | "
         f"MAE={final_result['mae']:.4f}")