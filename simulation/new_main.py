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
import numpy as np

import pandas as pd
from typing import List, Tuple, Dict, Optional
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
NUM_EPOCHS       = 100
LR               = 1e-3
LAMBDA           = 1.0
memory_capacity  = 5000
VAL_SIZE         = 0.2     # train 내부에서 val로 떼어낼 비율
TEST_SIZE        = 0.2     # 전체 데이터에서 test로 떼어낼 비율

USE_REPLAY = True
USE_LARS   = True
USE_SCALING = True
                                                                            
date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M")

# ── 장치 설정 ─────────────────────────────────────────────────────
device = get_device()

# ── 리플레이 버퍼 ─────────────────────────────────────────────────
#     [요청 2] 버퍼에는 항상 raw(스케일링되지 않은) feature 값만 저장합니다.
#     스케일러가 태스크마다 새로 fit되므로, 버퍼에 "그때그때 스케일링된 값"을
#     저장해두면 서로 다른 기준으로 정규화된 값들이 뒤섞이게 됩니다.
#     raw로 저장해두고, 사용 시점(매 태스크)의 스케일러로 그때그때
#     즉석 변환해서 사용하는 방식으로 이 문제를 피합니다.
memory_x:       List[torch.Tensor] = []   # raw feature
memory_y:       List[torch.Tensor] = []   # target (스케일링 대상 아님)
memory_teacher: List[torch.Tensor] = []   # teacher 예측값 (target-space, 스케일링 대상 아님)
memory_loss:    List[float]        = []
seen_examples = 0

# ── 가장 최근에 fit된 스케일러 (매 태스크 학습 시작 시 갱신됨) ──────
#     [요청 3] 모든 태스크 학습이 끝난 시점에 이 변수에 담겨있는 스케일러가
#     곧 "마지막 태스크의 current+replay raw로 fit된 최종 스케일러"이며,
#     이 스케일러로 test 데이터를 딱 한 번 변환한 뒤 최종 평가에 사용합니다.
CURRENT_SCALER: Optional[MinMaxScaler] = None

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
freq = [3_500_000_000]
MBC_CheongJu = [Transmitter("MBC-CheongJu", 127.433977355451, 36.61907632039761)]
Broad_CheongJu = [Transmitter('Broad-CheongJu', 127.47905459727248, 36.63418678335862)]
MBC_DaeJeon = [Transmitter("MBC-DaeJeon", 127.397198121239, 36.3760857047283)]
KBS_DaeJeon = [Transmitter("KBS-DaeJeon", 127.380567392303, 36.3704437169546)]
CMB_DaeJeon = [Transmitter("CMB-DaeJeon", 127.419676653034, 36.3341326962576)]
MBC_ChungJu = [Transmitter("MBC-ChungJu", 127.924378514041, 36.9585291745712)]
KBS_ChungJu = [Transmitter("KBS-ChungJu", 127.920483843397, 36.9724980330778)]
KBS_main = [Transmitter("KBS_main", 126.916716838156, 37.5259698897016)]
SBS = [Transmitter("SBS", 126.87374657727, 37.5291902429029)]
MBC_sa = [Transmitter("MBC_sa", 126.890988995582, 37.5811234199086)]

KBS_KangNeung = [Transmitter("KBS-KangNeung", 128.891256884067, 37.7520385140085)]
MBC_KangNeung = [Transmitter("MBC-KangNeung", 128.904230376246, 37.7709174571674)]
TBN_DaeGu = [Transmitter("TBN-DaeGu", 128.580363461352, 35.843420581465)]
TBC_DaeGu = [Transmitter("TBC-DaeGu", 128.622440960025, 35.8323075963103)]

JeonJu_Radio = [Transmitter("JeonJu Radio", 127.158316016663, 35.8489105505397)]
KBS_JeonJu = [Transmitter("KBS-JeonJu", 127.104790219087, 35.8221763698401)]
KBS_MokPo = [Transmitter("KBS-MokPo", 126.394348103215, 34.8120944953763)]
KBS_SunCheon = [Transmitter("KBS-SunCheon", 127.48480049784, 34.9683819549619)]
KBS_GwangJu = [Transmitter("KBS-GwangJu", 126.854624825405, 35.1581118750609)]



task_configs: List[Tuple] = [
    (MBC_CheongJu, freq, "MBC_CheongJu.csv"),
    (Broad_CheongJu, freq, "Broad_CheongJu.csv"),
    (MBC_DaeJeon, freq, "MBC_DaeJeon.csv"),
    (KBS_DaeJeon, freq, "KBS_DaeJeon.csv"),
    (CMB_DaeJeon, freq, "CMB_DaeJeon.csv"),
    (MBC_ChungJu, freq, "MBC_ChungJu.csv"),
    (KBS_ChungJu, freq, "KBS_ChungJu.csv"),
    (KBS_main, freq, "KBS_main.csv"),
    (SBS, freq, "SBS.csv"),
    (MBC_sa, freq, "MBC_sa.csv"),
    (KBS_KangNeung, freq, "KBS_KangNeung.csv"),
    (MBC_KangNeung, freq, "MBC_KangNeung.csv"),
    (TBN_DaeGu, freq, "TBN_DaeGu.csv"),
    (TBC_DaeGu, freq, "TBC_DaeGu.csv"),
    (JeonJu_Radio, freq, "JeonJu_Radio.csv"),
    (KBS_JeonJu, freq, "KBS_JeonJu.csv"),
    (KBS_MokPo, freq, "KBS_MokPo.csv"),
    (KBS_SunCheon, freq, "KBS_SunCheon.csv"),
    (KBS_GwangJu, freq, "KBS_GwangJu.csv")
]

# ── 저장 경로 ─────────────────────────────────────────────────────
save_model_dir  = "simulation/ER_/model/"
plot_dir        = os.path.join(save_model_dir, "plots")
history_dir     = os.path.join(save_model_dir, "history")
scaler_dir      = os.path.join(save_model_dir, "scalers")
save_model_path = os.path.join(save_model_dir, "model.pth")
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)


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


def add_buffer(x_raw, y, t, loss: float):
    """
    [요청 2] x_raw는 반드시 스케일링되지 않은 raw feature 값이어야 합니다.
    (호출부인 train_epoch에서 X_raw를 넘겨주도록 되어 있습니다.)
    """
    global seen_examples, memory_x, memory_y, memory_teacher, memory_loss
    seen_examples += 1

    if len(memory_x) < memory_capacity:
        memory_x.append(x_raw)
        memory_y.append(y)
        memory_teacher.append(t)
        memory_loss.append(loss)
        return

    j = random.randint(0, seen_examples - 1)
    if j < memory_capacity:
        victim = lars_victim() if USE_LARS else j
        memory_x[victim]       = x_raw
        memory_y[victim]       = y
        memory_teacher[victim] = t
        memory_loss[victim]    = loss


# ════════════════════════════════════════════════════════════════════
# [요청 1] 스케일링 관련 함수
# ════════════════════════════════════════════════════════════════════
def fit_task_scaler(current_raw_features: np.ndarray) -> Optional[MinMaxScaler]:
    """
    이번 태스크의 current raw feature와, 이 시점 리플레이 버퍼에 쌓여있는
    replay raw feature를 하나로 합쳐서 MinMaxScaler를 fit합니다.

    - use_scaling=False 이면 스케일링을 전혀 적용하지 않고 None을 반환합니다.
      (None이 전달되면 이후 모든 transform 지점에서 raw 값을 그대로 사용합니다.)
    - 버퍼가 비어있는 첫 태스크에서는 current raw만으로 fit됩니다.
    """
    if not USE_SCALING:
        return None

    combined = current_raw_features
    if len(memory_x) > 0:
        buf_np = torch.stack(memory_x).cpu().numpy()
        combined = np.concatenate([combined, buf_np], axis=0)

    scaler = MinMaxScaler()
    scaler.fit(combined)
    return scaler


def apply_scaler(raw_array: np.ndarray, scaler: Optional[MinMaxScaler]) -> np.ndarray:
    """scaler가 None이면 raw 그대로, 아니면 transform 결과를 반환합니다."""
    if scaler is None:
        return raw_array
    return scaler.transform(raw_array).astype("float32")


# ════════════════════════════════════════════════════════════════════
# Dataset 정의
# ════════════════════════════════════════════════════════════════════
class tensor_dataset(Dataset):
    """
    DataFrame(raw) → (X_model, y, X_raw) 텐서로 변환.

    - X_raw   : scaler와 무관하게 항상 원본(raw) feature 값
    - X_model : scaler가 주어지면 transform된 값(모델 입력용),
                scaler가 None이면(=스케일링 비활성) X_raw와 동일
    """
    def __init__(self, data: pd.DataFrame, scaler: Optional[MinMaxScaler] = None):
        X_raw   = data[features].values.astype("float32")
        X_model = apply_scaler(X_raw, scaler)

        self.X_model = torch.tensor(X_model, dtype=torch.float32)
        self.X_raw   = torch.tensor(X_raw,   dtype=torch.float32)
        self.y       = torch.tensor(data[target].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_model)

    def __getitem__(self, idx):
        return self.X_model[idx], self.y[idx], self.X_raw[idx]


class replay_dataset(Dataset):
    """
    current_ds + replay_ds 혼합 Dataset.
    두 갈래 모두 (x_model, y, y_teacher, is_replay, x_raw) 5-tuple로 통일합니다.
    """
    def __init__(self, current_ds: Dataset, replay_ds: Dataset = None):
        self.cur     = current_ds
        self.rep     = replay_ds
        self.cur_len = len(current_ds)
        self.rep_len = len(replay_ds) if replay_ds is not None else 0

    def __len__(self):
        return self.cur_len + self.rep_len

    def __getitem__(self, idx):
        if idx < self.cur_len:
            x_model, y, x_raw = self.cur[idx]
            return x_model, y, torch.zeros_like(y), False, x_raw   # is_replay=False
        else:
            x_model, y, y_t, x_raw = self.rep[idx - self.cur_len]
            return x_model, y, y_t, True, x_raw                    # is_replay=True


# ════════════════════════════════════════════════════════════════════
# DataLoader 빌더
# ════════════════════════════════════════════════════════════════════
def build_dataloader(task_ds_raw: pd.DataFrame, scaler: Optional[MinMaxScaler]) -> DataLoader:
    """
    학습용: current(raw→scaler로 변환) + replay(raw→**같은** scaler로 변환) 혼합 DataLoader.

    [요청 1] current와 replay 모두 "이번 태스크를 위해 새로 fit한 동일한 scaler"로
             변환되어 같은 배치 안에서 스케일 기준이 일치합니다.

    모든 텐서를 CPU에 둔 채로 Dataset을 구성하고, 실제 device 이동은
    train_epoch에서 배치 단위로 한 번에 수행합니다. (current/replay 텐서가
    서로 다른 device에 있는 상태로 배치가 섞여 collate에 실패하는 문제를
    원천적으로 방지하기 위함입니다.)
    """
    cur_ds    = tensor_dataset(task_ds_raw, scaler)
    replay_ds = None

    if len(memory_x) > 0:
        x_buf_raw = torch.stack(memory_x)                        # CPU, raw
        x_buf_model_np = apply_scaler(x_buf_raw.numpy(), scaler)  # 이번 태스크의 scaler로 변환
        x_buf_model = torch.tensor(x_buf_model_np, dtype=torch.float32)

        y_buf = torch.stack(memory_y)
        t_buf = torch.stack(memory_teacher)

        replay_ds = TensorDataset(x_buf_model, y_buf, t_buf, x_buf_raw)
        print(f"  Replay buffer size: {len(replay_ds)} "
             f"(raw로 저장되어 있으며, 이번 태스크의 scaler로 매번 재변환됨)")
    else:
        print("  Replay buffer is empty.")

    full_ds = replay_dataset(cur_ds, replay_ds)
    return DataLoader(full_ds, batch_size=BATCH_SIZE,
                      shuffle=True, pin_memory=False)


def val_dataloader(task_ds_raw: pd.DataFrame, scaler: Optional[MinMaxScaler]) -> DataLoader:
    """
    검증/평가용: replay 없이 순수 데이터만 담은 DataLoader.
    - train_on_task 내부에서 매 epoch validation loss 계산용 (scaler는 transform만, fit 없음)
    - 최종 test 평가(evaluate)에서도 재사용 (scaler=최종 스케일러)
    """
    cur_ds  = tensor_dataset(task_ds_raw, scaler)
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

    for X, Y, Y_t, is_rep, X_raw in data_loader:
        X, Y, Y_t, is_rep, X_raw = (X.to(device), Y.to(device),
                                    Y_t.to(device), is_rep.to(device),
                                    X_raw.to(device))
        optimizer.zero_grad()
        Y_pred = model(X)   # ← 모델 입력은 항상 X(스케일링된 값, 또는 스케일링 비활성 시 raw)

        cur_loss_per_sample = lfn.MSE_loss_per_sample(Y_pred, Y)  # (B,)
        # ── [버그 수정] current + replay 전체를 "정답(Y)" 기준으로 지도학습 ──
        loss_batch = lfn.MSE_loss(Y_pred, Y)

        # ── [버그 수정] rep_loss는 반드시 replay 샘플만 골라서 계산해야 합니다.
        #    Y_t는 current 샘플에서 torch.zeros_like(y)로 채워진 자리표시자이므로,
        #    마스킹 없이 배치 전체로 계산하면 current 샘플의 예측이 이유 없이
        #    0으로 끌려가는 오염된 gradient가 발생합니다. (버퍼가 채워지는
        #    두 번째 태스크부터 이 문제가 나타나 loss가 튀어 오르고,
        #    이후 특정 수준에서 정체되는 원인이었습니다.)
        if USE_REPLAY and is_rep.any():
            rep_loss = lfn.MSE_loss(Y_pred[is_rep], Y_t[is_rep])   # replay 샘플만 선택
            total_loss_batch = loss_batch + LAMBDA * rep_loss
        else:
            total_loss_batch = loss_batch

        teacher = Y_pred.detach().clone()

        total_loss_batch.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for i in range(X.size(0)):
            add_buffer(
                X_raw[i].detach().cpu(),
                Y[i].detach().cpu(),
                teacher[i].detach().cpu(),
                cur_loss_per_sample[i].item()
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

    for X, Y, _, _, _ in loader:
        X, Y   = X.to(device), Y.to(device)
        Y_pred = model(X)
        total_loss += lfn.MSE_loss(Y_pred, Y).item()

    return total_loss / max(n_batches, 1)


# ════════════════════════════════════════════════════════════════════
# [수정] 학습 순서에 따른 "연속" train/val loss curve 시각화
# ════════════════════════════════════════════════════════════════════
#     기존에는 태스크(지역)별로 각각 별도의 PNG 이미지를 생성했지만,
#     이제는 지역별 개별 이미지를 만들지 않고, 모든 태스크를 학습한
#     "순서 그대로" epoch을 이어붙여 하나의 그래프에 담습니다.
#
#     예) Task A가 200epoch, Task B가 200epoch이라면
#         global_epoch 1~200   → Task A
#         global_epoch 201~400 → Task B
#     로 이어지는 하나의 연속된 x축을 사용하고, 각 태스크의 경계에는
#     점선과 태스크 이름을 표시해 "어디서부터 어느 태스크인지" 구분만
#     시각적으로 돕습니다. (별도 이미지 파일을 만드는 것이 아니라
#     같은 하나의 그래프 안에 표시만 되는 것입니다.)
def plot_continuous_training_curve(history_all: Dict[str, Dict[str, List[float]]]):
    """
    history_all 은 태스크가 학습된 순서대로 삽입되므로 (Python dict는
    삽입 순서를 보존합니다), 이 순서를 그대로 global_epoch 축으로
    이어붙여 하나의 연속 곡선을 그립니다.
    """
    task_names = list(history_all.keys())
    if not task_names:
        return

    all_train, all_val, all_task_col, all_epoch_in_task = [], [], [], []
    boundaries = []   # 각 태스크가 시작되는 global_epoch 위치 (경계선 표시용)
    cursor = 0

    for task_name in task_names:
        train_losses = history_all[task_name]["train"]
        val_losses   = history_all[task_name]["val"]

        boundaries.append((cursor, task_name))
        for i, (tr, vl) in enumerate(zip(train_losses, val_losses), start=1):
            all_train.append(tr)
            all_val.append(vl)
            all_task_col.append(task_name)
            all_epoch_in_task.append(i)
        cursor += len(train_losses)

    global_epoch = list(range(1, len(all_train) + 1))

    # ── 그래프 (지역별로 나누지 않고 전체 학습 순서 하나로) ──────────
    plt.figure(figsize=(max(10, len(global_epoch) / 40), 6))
    plt.plot(global_epoch, all_train, label=L("Train Loss", "Train Loss"),
             color="steelblue", linewidth=1)
    plt.plot(global_epoch, all_val, label=L("Val Loss", "Val Loss"),
             color="orangered", linewidth=1)

    # 태스크 경계 표시 (동일 그래프 내 보조선/라벨일 뿐, 별도 이미지 아님)
    for start_idx, task_name in boundaries:
        plt.axvline(x=start_idx + 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        plt.text(start_idx + 1, plt.ylim()[1], task_name.replace(".csv", ""),
                 rotation=90, fontsize=6, va='top', ha='left', alpha=0.7)

    plt.xlabel(L("학습 순서에 따른 누적 Epoch", "Cumulative Epoch (training order)"))
    plt.ylabel(L("MSE Loss", "MSE Loss"))
    plt.title(L("전체 태스크 학습 순서에 따른 Train / Val Loss",
                "Train / Val Loss over Continual Training Order"))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(plot_dir, f"continuous_training_curve_{date}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[그래프] 학습 순서에 따른 연속 loss curve 저장 → {path}")

    # 수치 기록은 이미지가 아니므로 함께 저장 (재분석용, 지역 구분 컬럼 포함)
    hist_df = pd.DataFrame({
        "global_epoch":  global_epoch,
        "task":          all_task_col,
        "epoch_in_task": all_epoch_in_task,
        "train_loss":    all_train,
        "val_loss":      all_val,
    })
    hist_csv = os.path.join(history_dir, f"continuous_training_curve_{date}.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"[기록] 학습 순서에 따른 loss 기록 저장 → {hist_csv}")


# ════════════════════════════════════════════════════════════════════
# [복원] 통합 test set 기준 continual-learning 성능 곡선 시각화
# ════════════════════════════════════════════════════════════════════
def plot_continual_test_curve(records: List[Dict]):
    """
    records: [{"stage": 0, "stage_task": "...", "mse":.., "rmse":.., "mae":..}, ...]

    태스크를 하나씩 학습해 나가면서, 그 태스크 학습 직후 "그 태스크에서
    막 fit된 scaler"로 test 데이터를 변환해 측정한 MSE/RMSE 변화를
    하나의 곡선으로 그립니다.
    """
    if not records:
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(df["stage"], df["mse"], marker='o', color="steelblue")
    axes[0].set_xticks(df["stage"])
    axes[0].set_xticklabels(df["stage_task"], rotation=30, ha='right', fontsize=7)
    axes[0].set_ylabel("MSE")
    axes[0].set_title(L("Test Set 기준 MSE 변화 (매 태스크 자신의 scaler로 평가)",
                        "MSE on Test Set (evaluated with each stage's own scaler)"))
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["stage"], df["rmse"], marker='o', color="orangered")
    axes[1].set_xticks(df["stage"])
    axes[1].set_xticklabels(df["stage_task"], rotation=30, ha='right', fontsize=7)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title(L("Test Set 기준 RMSE 변화 (매 태스크 자신의 scaler로 평가)",
                        "RMSE on Test Set (evaluated with each stage's own scaler)"))
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
# 태스크 학습
# ════════════════════════════════════════════════════════════════════
def train_on_task(train_ds_raw: pd.DataFrame, task_name: str, val_size: float = VAL_SIZE):
    """
    train_ds_raw (해당 태스크의 raw 학습 데이터)를 train/val로 분리한 뒤,

        [요청 1] "이번 태스크의 raw train" + "현재 buffer의 raw replay"를
                 합쳐서 이번 태스크 전용 스케일러를 새로 fit합니다.

    - train : 실제 파라미터 업데이트 + 리플레이 버퍼 채우기에 사용
              (current/replay 모두 이 스케일러로 변환된 값이 모델에 들어감)
    - val   : 매 epoch 학습 이후 성능 확인(검증)에만 사용, 학습에는 관여하지 않음
              (같은 스케일러로 transform만 적용, fit에는 사용하지 않음 → leakage 방지)
    """
    global CURRENT_SCALER

    optimizer = optim.Adam(model.parameters(), lr=LR)
    sched     = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── train 내부에서 train/val 분리 (raw 상태) ────────────────────
    inner_train, inner_val = train_test_split(
        train_ds_raw, test_size=val_size, shuffle=True, random_state=918
    )
    inner_train = inner_train.reset_index(drop=True)
    inner_val   = inner_val.reset_index(drop=True)

    # ── [요청 1] 이번 태스크의 스케일러: current(inner_train) + replay(buffer) ──
    task_scaler = fit_task_scaler(inner_train[features].values.astype("float32"))
    CURRENT_SCALER = task_scaler   # 매 태스크마다 갱신 → 마지막 호출 후엔 "최종 스케일러"가 됨

    if task_scaler is not None:
        joblib.dump(task_scaler,
                   os.path.join(scaler_dir, f"{task_name.replace('.csv','')}_scaler.pkl"))

    print(f"\n=== Task: {task_name} ===")
    print(f"  inner_train={len(inner_train)}, inner_val={len(inner_val)}, "
         f"use_scaling={USE_SCALING}")

    # 학습용 로더: (inner_train + replay buffer) 혼합, 둘 다 task_scaler로 변환
    train_loader = build_dataloader(inner_train, task_scaler)
    # 검증용 로더: inner_val만 (replay 없음, task_scaler로 transform만 적용)
    val_loader   = val_dataloader(inner_val, task_scaler)

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

    # ── 학습 종료 후, 이 태스크의 best val_loss 체크포인트를 다시 불러와
    #    in-memory model을 최선 상태로 되돌립니다.
    load_model(model, save_model_path)

    # ── 태스크별 이력 기록 (지역별 개별 이미지는 생성하지 않고, 수치만
    #    history_all에 순서대로 쌓아둡니다. 실제 그래프는 main()에서
    #    모든 태스크 학습이 끝난 뒤 plot_continuous_training_curve()로
    #    "학습 순서에 따른 하나의 연속 곡선"으로 한 번에 그립니다.) ──
    history_all[task_name] = {"train": train_losses, "val": val_losses}


# ════════════════════════════════════════════════════════════════════
# 평가 함수
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(task_name: str, test_ds_raw: pd.DataFrame,
            scaler: Optional[MinMaxScaler] = None):
    """
    현재 시점의 model을 test_ds_raw(raw 상태)로 평가합니다.

    [수정] test 데이터의 스케일링은 이 함수 내부(tensor_dataset)에서
    scaler 인자로 수행됩니다. main()에서는 "매 태스크 학습이 끝난 직후,
    그 태스크에서 막 fit된 scaler(해당 stage의 CURRENT_SCALER)"를 이
    scaler 인자로 넘겨, 태스크가 하나씩 추가될 때마다 그 시점 기준으로
    test 성능을 즉시 측정합니다.
    """
    loader = val_dataloader(test_ds_raw, scaler)
    model.eval()

    total_mse  = 0.0
    total_mae  = 0.0
    n_batches  = len(loader)

    for X, Y, _, _, _ in loader:
        X, Y   = X.to(device), Y.to(device)
        Y_pred = model(X)
        total_mse += lfn.MSE_loss(Y_pred, Y).item()
        total_mae += (Y_pred - Y).abs().mean().item()

    avg_mse  = total_mse / n_batches
    avg_rmse = avg_mse ** 0.5
    avg_mae  = total_mae / n_batches

    print(f"\n[최종 Test 평가] {task_name}  (use_scaling={USE_SCALING})")
    print(f"  MSE  : {avg_mse:.4f}")
    print(f"  RMSE : {avg_rmse:.4f}")
    print(f"  MAE  : {avg_mae:.4f}")

    return {'mse': avg_mse, 'rmse': avg_rmse, 'mae': avg_mae}


# ════════════════════════════════════════════════════════════════════
# 데이터 분리 (전체 데이터 → train 전체 / test)
# ════════════════════════════════════════════════════════════════════
def data_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    DataFrame을 train(전체) / test 로 분리합니다. (raw 상태, 스케일링 없음)

    - train : train_on_task() 내부에서 다시 train/val로 나뉘어 사용됩니다.
    - test  : 모든 태스크 학습이 끝난 뒤, 최종 스케일러로 딱 한 번
              변환되어 최종 평가에만 사용됩니다.
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
            print(f"\n[CSV 로드] {file_name} 이미 존재 → 로드합니다.")
            df = pd.read_csv(save_path, index_col=0)
            df = df.reset_index(drop=True)
            print(f"[CSV 로드] {file_name} 완료 ({len(df)}행)")
        else:
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

    # ── 2. 각 태스크를 train(전체) / test로 분리 (raw 상태 유지) ─────
    #     스케일링은 여기서 하지 않습니다. train은 train_on_task 내부에서,
    #     test는 모든 학습이 끝난 뒤 딱 한 번 스케일링됩니다.
    raw_train: Dict[str, pd.DataFrame] = {}
    raw_test:  Dict[str, pd.DataFrame] = {}
    for file_name, df in tasks:
        train_data, test_data = data_split(df)
        raw_train[file_name] = train_data
        raw_test[file_name]  = test_data
        print(f"  {file_name}: train(total)={len(train_data)}, test={len(test_data)}")

    # ── 3. 통합 test set은 미리 하나로 합쳐둡니다 (raw 상태) ─────────
    #     [요청 3] 실제 스케일링은 매 태스크 학습 직후, 그 태스크에서
    #     막 fit된 scaler로 매번 다시 수행됩니다 (여기서는 raw로만 합침).
    if False:
        combined_test_raw = pd.concat(list(raw_test.values()), ignore_index=True)
        print(f"[통합 Test Set] 전체 태스크 test를 합쳐 총 {len(combined_test_raw)}행 구성 (raw 상태)")

        # ── 4. 태스크 순서대로 ER 학습 + 매 태스크 직후 test 평가 ────────
        #     [요청 1,2] 스케일링은 train_on_task 내부에서 매 태스크마다
        #     (current raw + replay raw)로 새로 fit되어 적용됩니다.
        #     [요청 3] 그 태스크 학습이 끝나는 즉시, 방금 fit된 scaler로
        #     통합 test set을 변환해 성능을 측정하고 기록합니다.
        print(f"\n=== Experience Replay 학습 시작 (use_scaling={USE_SCALING}) ===")
        continual_eval_records: List[Dict] = []

        for stage, (file_name, train_data) in enumerate(raw_train.items()):
            train_on_task(train_data, task_name=file_name, val_size=VAL_SIZE)

            # 이번 태스크 학습 중 fit_task_scaler()가 만든, 이 태스크 전용 scaler
            stage_scaler = CURRENT_SCALER
            stage_result = evaluate(f"stage{stage}_{file_name}",
                                    combined_test_raw, scaler=stage_scaler)
            continual_eval_records.append({
                "stage":      stage,
                "stage_task": file_name,
                "mse":        stage_result["mse"],
                "rmse":       stage_result["rmse"],
                "mae":        stage_result["mae"],
            })
    print(f"\n=== Experience Replay 학습 시작 (use_scaling={USE_SCALING}) ===")
    continual_eval_records: List[Dict] = []
    seen_test_list: List[pd.DataFrame] = []   # ← 학습이 끝난 도시의 test만 누적

    for stage, (file_name, train_data) in enumerate(raw_train.items()):
        train_on_task(train_data, task_name=file_name, val_size=VAL_SIZE)

        # 방금 학습을 마친 태스크의 test를 누적 목록에 추가
        seen_test_list.append(raw_test[file_name])
        seen_only_test = pd.concat(seen_test_list, ignore_index=True)  # "지금까지 배운 도시"만

        stage_scaler = CURRENT_SCALER
        stage_result = evaluate(f"stage{stage}_{file_name}",
                                seen_only_test, scaler=stage_scaler)   # ← combined_test_raw 대신 seen_only_test
        continual_eval_records.append({
            "stage":      stage,
            "stage_task": file_name,
            "n_seen_tasks": len(seen_test_list),   # 참고용: 이 시점 test에 포함된 도시 수
            "mse":        stage_result["mse"],
            "rmse":       stage_result["rmse"],
            "mae":        stage_result["mae"],
        })

    # ── 5. 학습 순서에 따른 연속 train/val loss curve 시각화 ─────────
    #     (지역별 개별 이미지는 생성하지 않고, 하나의 연속된 그래프만 저장)
    plot_continuous_training_curve(history_all)

    # ── 6. 매 태스크 직후 측정한 test 성능을 하나의 곡선으로 시각화 ────
    plot_continual_test_curve(continual_eval_records)

    # ── 7. 최종 모델 저장 ─────────────────────────────────────────
    final_path = os.path.join(save_model_dir, f"model_{date}.pth")
    save_model(model, final_path, epoch=NUM_EPOCHS, loss=None)

    # ── 8. 전체 stage 결과 CSV 저장 (스케일링 유무 비교가 쉽도록 flag를 파일명에 포함) ──
    final_result = continual_eval_records[-1]
    csv_path = os.path.join(save_model_dir, f"results_{date}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['use_scaling', 'Stage', 'Task', 'MSE', 'RMSE', 'MAE'])
        for rec in continual_eval_records:
            writer.writerow([USE_SCALING, rec['stage'], rec['stage_task'],
                             f"{rec['mse']:.4f}",
                             f"{rec['rmse']:.4f}",
                             f"{rec['mae']:.4f}"])
    print(f"\n결과 저장 완료 → {csv_path}")
    print(f"최종(전체 학습 완료 후) test set 성능 (use_scaling={USE_SCALING}): "
         f"MSE={final_result['mse']:.4f} | "
         f"RMSE={final_result['rmse']:.4f} | "
         f"MAE={final_result['mae']:.4f}")