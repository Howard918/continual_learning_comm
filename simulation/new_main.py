import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np

import pandas as pd
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use("Agg")   # 화면 없는 환경(서버)에서도 저장 가능하도록
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

import ER_.loss_func as lfn
from ER_.utils import device as get_device
from csv_factory import create_CSV
from environments.transmitter import Transmitter


# ── 재현성 설정 ───────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    random, numpy, torch(CPU/CUDA/MPS)의 난수 생성기를 모두 동일한 seed로
    고정합니다. cudnn의 비결정적 알고리즘 선택도 함께 비활성화하여,
    동일한 코드와 데이터로 재실행했을 때 항상 같은 결과가 나오도록 합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


set_seed(SEED)

# DataLoader의 shuffle(=torch.randperm 기반)에 사용할 고정 seed 생성기.
# 이 generator를 shuffle=True인 DataLoader마다 전달하여, 배치 구성 순서를
# 실행마다 동일하게 재현합니다.
dataloader_generator = torch.Generator()
dataloader_generator.manual_seed(SEED)


# ── 하이퍼파라미터 ────────────────────────────────────────────────
BATCH_SIZE       = 256
NUM_EPOCHS       = 100
LR               = 1e-3
LAMBDA           = 0.5
BETA             = 0
memory_capacity  = 5000
VAL_SIZE         = 0.2     # train 내부에서 val로 떼어낼 비율
TEST_SIZE        = 0.2     # 전체 데이터에서 test로 떼어낼 비율

USE_REPLAY  = False
USE_LARS    = False
USE_SCALING = True

save = "Normal_MLP"

# ── 장치 설정 ─────────────────────────────────────────────────────
device = get_device()

# ── 리플레이 버퍼 ─────────────────────────────────────────────────
# 학습에 사용되는 current/replay 샘플을 저장하는 버퍼입니다.
# raw(스케일링 전) feature로 저장해두고, 사용 시점의 스케일러로 변환합니다.
memory_x:       List[torch.Tensor] = []   # raw feature
memory_y:       List[torch.Tensor] = []   # target (스케일링 대상 아님)
memory_teacher: List[torch.Tensor] = []   # teacher 예측값 (target-space, 스케일링 대상 아님)
memory_loss:    List[float]        = []
seen_examples = 0

# ── 검증 전용 누적 버퍼 ───────────────────────────────────────────
# 매 태스크의 inner_val을 이 버퍼에 계속 누적하되, 고정 크기를 넘으면
# reservoir sampling으로 기존 항목을 교체합니다. train replay buffer와
# 완전히 분리된 별도 풀이며, 이 버퍼의 내용은 학습(gradient 계산)에
# 절대 사용되지 않고 오직 validation loss 계산에만 쓰입니다.
val_memory_x: List[torch.Tensor] = []
val_memory_y: List[torch.Tensor] = []
val_seen_examples = 0
VAL_MEMORY_CAPACITY = 2000   # 검증 풀의 고정 크기

# 가장 최근에 fit된 스케일러. 매 태스크 학습 시작 시 갱신되므로,
# 모든 태스크 학습이 끝난 시점에는 마지막 태스크의 스케일러가 남습니다.
CURRENT_SCALER: Optional[RobustScaler] = None

# 태스크별 epoch 학습 이력(train/val loss)을 순서대로 기록합니다.
# history_all[task_name] = {"train": [...], "val": [...]}
history_all: Dict[str, Dict[str, List[float]]] = {}

# ── 데이터 설정 ───────────────────────────────────────────────────
default_data_path = "MLP/DATA/"
features          = ["R", "D", "H", "F"]
target            = "PL"

# ── CSV 생성 설정 ─────────────────────────────────────────────────
# (transmitter 그룹, 주파수 리스트, 저장 파일명) 형태로 태스크를 정의합니다.
# 태스크를 추가하려면 아래 리스트에 항목을 추가하면 됩니다.
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
    모델 파라미터와 학습 상태를 저장합니다.
        - model_state_dict : 모델 파라미터
        - epoch            : 저장 시점의 epoch (선택)
        - optimizer_state  : optimizer 상태 (선택, 이어서 학습 시 필요)
        - loss             : 저장 시점의 val_loss (선택)
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
    저장된 체크포인트를 불러와 모델(및 optimizer)에 적용합니다.

    Returns:
        start_epoch (int): 저장 시점의 epoch 번호
        loss        (float): 저장 시점의 val_loss
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
# 리플레이 버퍼 관련 함수
# ════════════════════════════════════════════════════════════════════
def lars_victim() -> int:
    """버퍼 내 loss가 낮은(=쉬운) 샘플일수록 교체될 확률이 높도록 victim 인덱스를 뽑습니다."""
    losses = torch.tensor(memory_loss, dtype=torch.float)
    inv    = 1.0 / (losses + 1e-8)
    prob   = inv / inv.sum()
    return torch.multinomial(prob, 1).item()


def add_buffer(x_raw, y, t, loss: float):
    """
    reservoir sampling(또는 LARS)으로 리플레이 버퍼에 raw 샘플 하나를 추가합니다.
    버퍼가 가득 찬 이후에는 확률적으로 기존 샘플을 교체합니다.
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
# 검증 전용 누적 버퍼 관련 함수
# ════════════════════════════════════════════════════════════════════
def add_val_buffer(x_raw: torch.Tensor, y: torch.Tensor):
    """
    검증 전용 reservoir buffer에 raw 샘플 하나를 추가합니다.
    이 버퍼는 학습(gradient 계산)에 사용되지 않고 오직 validation loss
    계산에만 쓰입니다.
    """
    global val_seen_examples, val_memory_x, val_memory_y
    val_seen_examples += 1

    if len(val_memory_x) < VAL_MEMORY_CAPACITY:
        val_memory_x.append(x_raw)
        val_memory_y.append(y)
        return

    j = random.randint(0, val_seen_examples - 1)
    if j < VAL_MEMORY_CAPACITY:
        val_memory_x[j] = x_raw
        val_memory_y[j] = y


def accumulate_val_pool(inner_val_raw: pd.DataFrame):
    """
    이번 태스크의 inner_val(raw) 전체를 검증 누적 버퍼에 삽입합니다.
    태스크당 한 번만 호출되므로, 같은 샘플이 여러 epoch에 걸쳐
    반복 삽입되지 않습니다.
    """
    X_raw = inner_val_raw[features].values.astype("float32")
    Y_raw = inner_val_raw[target].values.astype("float32")
    for i in range(len(inner_val_raw)):
        add_val_buffer(
            torch.tensor(X_raw[i], dtype=torch.float32),
            torch.tensor([Y_raw[i]], dtype=torch.float32),
        )


def build_val_pool_df() -> pd.DataFrame:
    """
    검증 누적 버퍼(raw) 전체를 DataFrame으로 복원합니다.
    이 DataFrame은 val_dataloader(df, task_scaler)에 전달되어,
    이번 태스크의 스케일러로 일관되게 변환된 뒤 검증에 사용됩니다.
    """
    X_arr = torch.stack(val_memory_x).numpy()              # (N, len(features))
    Y_arr = torch.stack(val_memory_y).numpy().reshape(-1)  # (N,)
    df = pd.DataFrame(X_arr, columns=features)
    df[target] = Y_arr
    return df


# ════════════════════════════════════════════════════════════════════
# 스케일링 관련 함수
# ════════════════════════════════════════════════════════════════════
def fit_task_scaler(current_raw_features: np.ndarray) -> Optional[RobustScaler]:
    """
    이번 태스크의 current raw feature와, 이 시점 리플레이 버퍼에 쌓여있는
    replay raw feature를 하나로 합쳐서 RobustScaler를 fit합니다.

    - USE_SCALING=False이면 스케일링을 적용하지 않고 None을 반환합니다.
      (None이 전달되면 이후 모든 transform 지점에서 raw 값을 그대로 사용합니다.)
    - 버퍼가 비어있는 첫 태스크에서는 current raw만으로 fit됩니다.
    """
    if not USE_SCALING:
        return None

    combined = current_raw_features
    if len(memory_x) > 0:
        buf_np = torch.stack(memory_x).cpu().numpy()
        combined = np.concatenate([combined, buf_np], axis=0)

    scaler = RobustScaler()
    scaler.fit(combined)
    return scaler


def apply_scaler(raw_array: np.ndarray, scaler: Optional[RobustScaler]) -> np.ndarray:
    """scaler가 None이면 raw 값을 그대로, 아니면 transform 결과를 반환합니다."""
    if scaler is None:
        return raw_array
    return scaler.transform(raw_array).astype("float32")


# ════════════════════════════════════════════════════════════════════
# Dataset 정의
# ════════════════════════════════════════════════════════════════════
class tensor_dataset(Dataset):
    """
    DataFrame(raw) → (X_model, y, X_raw) 텐서로 변환합니다.

    - X_raw   : scaler와 무관하게 항상 원본(raw) feature 값
    - X_model : scaler가 주어지면 transform된 값(모델 입력용),
                scaler가 None이면(=스케일링 비활성) X_raw와 동일
    """
    def __init__(self, data: pd.DataFrame, scaler: Optional[RobustScaler] = None):
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
def build_dataloader(task_ds_raw: pd.DataFrame, scaler: Optional[RobustScaler]) -> DataLoader:
    """학습용: current(scaler로 변환) + replay(같은 scaler로 변환) 혼합 DataLoader."""
    cur_ds    = tensor_dataset(task_ds_raw, scaler)
    replay_ds = None

    if len(memory_x) > 0:
        x_buf_raw = torch.stack(memory_x)                        # CPU, raw
        x_buf_model_np = apply_scaler(x_buf_raw.numpy(), scaler)  # 이번 태스크의 scaler로 변환
        x_buf_model = torch.tensor(x_buf_model_np, dtype=torch.float32)

        y_buf = torch.stack(memory_y)
        t_buf = torch.stack(memory_teacher)

        replay_ds = TensorDataset(x_buf_model, y_buf, t_buf, x_buf_raw)
        print(f"  Replay buffer size: {len(replay_ds)}")
    else:
        print("  Replay buffer is empty.")

    full_ds = replay_dataset(cur_ds, replay_ds)
    return DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                      pin_memory=False, generator=dataloader_generator)


def val_dataloader(task_ds_raw: pd.DataFrame, scaler: Optional[RobustScaler]) -> DataLoader:
    """
    검증/평가용: replay 없이 순수 데이터만 담은 DataLoader.
    - train_on_task 내부에서 매 epoch validation loss 계산용 (scaler는 transform만, fit 없음)
    - 최종 test 평가(evaluate)에서도 재사용
    """
    cur_ds  = tensor_dataset(task_ds_raw, scaler)
    full_ds = replay_dataset(cur_ds)          # replay 없음
    return DataLoader(full_ds, batch_size=BATCH_SIZE,
                      shuffle=False, pin_memory=False)


# ════════════════════════════════════════════════════════════════════
# 학습 / 검증 함수
# ════════════════════════════════════════════════════════════════════
def train_epoch(data_loader, optimizer) -> float:
    """한 epoch 동안 (current + replay) 배치로 학습하고 평균 train loss를 반환합니다."""
    model.train()
    total_loss = 0.0

    for X, Y, Y_t, is_rep, X_raw in data_loader:
        X, Y, Y_t, is_rep, X_raw = (X.to(device), Y.to(device),
                                    Y_t.to(device), is_rep.to(device),
                                    X_raw.to(device))
        optimizer.zero_grad()
        Y_pred = model(X)

        cur_loss_per_sample = lfn.MSE_loss_per_sample(Y_pred, Y)  # (B,)
        is_cur = ~is_rep
        loss_batch_cur = lfn.MSE_loss(Y_pred[is_cur], Y[is_cur])

        if USE_REPLAY and is_rep.any():
            # current(정답) + replay(정답) + replay(teacher distillation)를
            # 각각 가중합하여 최종 loss를 구성합니다.
            loss_batch_rep = lfn.MSE_loss(Y_pred[is_rep], Y[is_rep])
            distill_loss   = lfn.MSE_loss(Y_pred[is_rep], Y_t[is_rep])
            total_loss_batch = (LAMBDA * loss_batch_cur
                               + (1 - LAMBDA) * loss_batch_rep
                               + BETA * distill_loss)
        else:
            total_loss_batch = loss_batch_cur

        teacher = Y_pred.detach().clone()

        total_loss_batch.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 리플레이 버퍼에는 이번 배치에서 새로 관측된 current 샘플만 추가합니다.
        for i in range(X.size(0)):
            if is_rep[i]:
                continue
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
    (매 epoch validation loss / 최종 test loss 계산에 공통으로 사용)
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
# 통합 test set 기준 continual-learning 성능 곡선 시각화
# ════════════════════════════════════════════════════════════════════
def plot_continual_test_curve(records: List[Dict]):
    """
    records: [{"stage": 0, "stage_task": "...", "mse":.., "rmse":.., "mae":..}, ...]

    태스크를 하나씩 학습해 나가면서, 그 태스크 학습 직후 해당 태스크에서
    막 fit된 scaler로 test 데이터를 변환해 측정한 MSE/RMSE 변화를
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
    path = os.path.join(plot_dir, f"test_curve_{save}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[그래프] 통합 test set 기준 continual-learning 곡선 저장 → {path}")

    csv_path = os.path.join(history_dir, f"test_curve_{save}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[기록] 통합 test set 평가 기록 저장 → {csv_path}")


# ════════════════════════════════════════════════════════════════════
# 태스크 학습
# ════════════════════════════════════════════════════════════════════
def train_on_task(train_ds_raw: pd.DataFrame, task_name: str, val_size: float = VAL_SIZE):
    """
    train_ds_raw(해당 태스크의 raw 학습 데이터)를 train/val로 분리한 뒤,
    "이번 태스크의 raw train" + "현재 buffer의 raw replay"를 합쳐서
    이번 태스크 전용 스케일러를 새로 fit합니다.

    - train : 실제 파라미터 업데이트 + 리플레이 버퍼 채우기에 사용
              (current/replay 모두 이 스케일러로 변환된 값이 모델에 들어감)
    - val   : 매 epoch 학습 이후 성능 확인(검증)에만 사용, 학습에는 관여하지 않음
              (같은 스케일러로 transform만 적용, fit에는 사용하지 않아 leakage를 방지)
              이번 태스크의 val은 검증 누적 버퍼에 추가되어, 과거 태스크의
              val과 함께 고정 크기 풀로 평가에 사용됩니다.
    """
    global CURRENT_SCALER

    optimizer = optim.Adam(model.parameters(), lr=LR)
    sched     = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    inner_train, inner_val = train_test_split(
        train_ds_raw, test_size=val_size, shuffle=True, random_state=SEED
    )
    inner_train = inner_train.reset_index(drop=True)
    inner_val   = inner_val.reset_index(drop=True)

    task_scaler = fit_task_scaler(inner_train[features].values.astype("float32"))
    CURRENT_SCALER = task_scaler   # 마지막 태스크 호출 후엔 최종 스케일러로 남음

    if task_scaler is not None:
        joblib.dump(task_scaler,
                   os.path.join(scaler_dir, f"{task_name.replace('.csv','')}_scaler.pkl"))

    print(f"\n=== Task: {task_name} ===")
    print(f"  inner_train={len(inner_train)}, inner_val(신규)={len(inner_val)}, "
         f"use_scaling={USE_SCALING}")

    accumulate_val_pool(inner_val)
    val_pool_df = build_val_pool_df()
    print(f"  검증 누적 풀 크기: {len(val_pool_df)} (최대 {VAL_MEMORY_CAPACITY}로 고정)")

    train_loader = build_dataloader(inner_train, task_scaler)
    val_loader   = val_dataloader(val_pool_df, task_scaler)

    best_val_loss = float('inf')
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for ep in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(train_loader, optimizer)
        sched.step()

        val_loss = compute_avg_loss(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {ep:>3}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} "
             f"| Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_model_path,
                      epoch=ep, optimizer=optimizer, loss=best_val_loss)

    # 이 태스크의 best val_loss 체크포인트를 다시 불러와 in-memory model을
    # 최선 상태로 되돌립니다.
    load_model(model, save_model_path)

    history_all[task_name] = {"train": train_losses, "val": val_losses}


# ════════════════════════════════════════════════════════════════════
# 평가 함수
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(task_name: str, test_ds_raw: pd.DataFrame,
            scaler: Optional[RobustScaler] = None):
    """
    현재 시점의 model을 test_ds_raw(raw 상태)로 평가합니다.
    test 데이터의 스케일링은 이 함수 내부(tensor_dataset)에서 scaler
    인자로 수행됩니다.
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

    print(f"\n[Test 평가] {task_name}  (use_scaling={USE_SCALING})")
    print(f"  MSE  : {avg_mse:.4f}")
    print(f"  RMSE : {avg_rmse:.4f}")
    print(f"  MAE  : {avg_mae:.4f}")

    return {'mse': avg_mse, 'rmse': avg_rmse, 'mae': avg_mae}


# ════════════════════════════════════════════════════════════════════
# 데이터 분리 (전체 데이터 → train 전체 / test)
# ════════════════════════════════════════════════════════════════════
def data_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    DataFrame을 train(전체) / test로 분리합니다. (raw 상태, 스케일링 없음)

    - train : train_on_task() 내부에서 다시 train/val로 나뉘어 사용됩니다.
    - test  : 매 태스크 학습 직후, 그 태스크의 스케일러로 변환되어
              평가에 사용됩니다.
    """
    train, test = train_test_split(df, test_size=test_size, shuffle=True, random_state=SEED)
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    return train, test


# ════════════════════════════════════════════════════════════════════
# CSV 생성 및 파일명 등록
# ════════════════════════════════════════════════════════════════════
def generate_csv_tasks(task_configs: List[Tuple]) -> List[Tuple[str, pd.DataFrame]]:
    """
    task_configs의 각 항목에 대해 CSV를 생성하고 (파일명, DataFrame) 리스트를
    반환합니다.

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

    print("=== CSV 데이터 생성 ===")
    tasks = generate_csv_tasks(task_configs)

    # 각 태스크를 train(전체) / test로 분리합니다. (raw 상태 유지)
    # 스케일링은 여기서 하지 않습니다. train은 train_on_task 내부에서,
    # test는 매 태스크 학습 직후 그 태스크의 스케일러로 변환됩니다.
    raw_train: Dict[str, pd.DataFrame] = {}
    raw_test:  Dict[str, pd.DataFrame] = {}
    for file_name, df in tasks:
        train_data, test_data = data_split(df)
        raw_train[file_name] = train_data
        raw_test[file_name]  = test_data
        print(f"  {file_name}: train(total)={len(train_data)}, test={len(test_data)}")

    print(f"\n=== Experience Replay 학습 시작 ===")
    continual_eval_records: List[Dict] = []
    seen_test_list: List[pd.DataFrame] = []   # 학습이 끝난 도시의 test만 누적

    for stage, (file_name, train_data) in enumerate(raw_train.items()):
        train_on_task(train_data, task_name=file_name, val_size=VAL_SIZE)

        seen_test_list.append(raw_test[file_name])
        seen_only_test = pd.concat(seen_test_list, ignore_index=True)   # 지금까지 배운 도시만
        whole_test = pd.concat(list(raw_test.values()), ignore_index=True)  # 전체 test (참고용)

        stage_scaler = CURRENT_SCALER
        stage_result = evaluate(f"stage{stage}_{file_name}",
                                whole_test, scaler=stage_scaler)
        continual_eval_records.append({
            "stage":        stage,
            "stage_task":   file_name,
            "n_seen_tasks": len(seen_test_list),
            "mse":          stage_result["mse"],
            "rmse":         stage_result["rmse"],
            "mae":          stage_result["mae"],
        })

    plot_continual_test_curve(continual_eval_records)

    final_path = os.path.join(save_model_dir, f"model_{save}.pth")
    save_model(model, final_path, epoch=NUM_EPOCHS, loss=None)

    final_result = continual_eval_records[-1]
    csv_path = os.path.join(save_model_dir, f"results_{save}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['use_scaling', 'Stage', 'Task', 'MSE', 'RMSE', 'MAE'])
        for rec in continual_eval_records:
            writer.writerow([USE_SCALING, rec['stage'], rec['stage_task'],
                             f"{rec['mse']:.4f}",
                             f"{rec['rmse']:.4f}",
                             f"{rec['mae']:.4f}"])
    print(f"\n결과 저장 완료 → {csv_path}")
    print(f"최종(전체 학습 완료 후) test set 성능: "
         f"MSE={final_result['mse']:.4f} | "
         f"RMSE={final_result['rmse']:.4f} | "
         f"MAE={final_result['mae']:.4f}")