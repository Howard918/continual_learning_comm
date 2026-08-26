"""
지역(송신소)별 CSV(R, D, H, PL, F 컬럼, 6개 주파수 대역이 하나로 합쳐진 형태)를 입력받아,
주파수 대역별로 독립된 MLP 회귀 모델을 학습하고, x축을 주파수 대역으로 하여
지역별 Test 성능(MSE/RMSE)을 비교하는 스크립트.

main.py의 모델/학습 설정(MyMLP 구조, RobustScaler, Adam+CosineAnnealingLR,
gradient clipping, SEED=42, train:val:test 분할 비율)을 그대로 재사용한다.
다만 여기서는 "동일 지역 내 주파수 대역 간 성능 비교"가 목적이므로,
지속 학습(ER)은 적용하지 않고 대역별로 독립적인(non-continual) 모델을 학습한다.

입력 CSV는 merge_by_region.py로 만든 형태를 가정한다:
    columns = [R, D, H, PL, F] (+ 있으면 freq_band 도 사용)
F 컬럼에 freq_band 라벨이 없는 경우, 시뮬레이션에 사용된 F 값(log10(Hz))이
대역마다 고유한 상수라는 점을 이용해 아래 FREQ_MAP으로 라벨을 복원한다.

사용 예:
    python train_freq_band_performance.py --data SBS_Seoul.csv
    python train_freq_band_performance.py --data-dir merged_by_region --epochs 60
"""

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

# ── main.py와 동일한 하이퍼파라미터 ───────────────────────────────────
SEED        = 42
BATCH_SIZE  = 256
NUM_EPOCHS  = 100
LR          = 1e-3
HIDDEN_DIM  = 64
LAYER_NUM   = 2
VAL_SIZE    = 0.2   # train 내부에서 val로 떼어낼 비율
TEST_SIZE   = 0.2   # 전체에서 test로 떼어낼 비율

# F(=log10(Hz)) 값 -> 주파수 대역 라벨. freq_band 컬럼이 없는 CSV를 위한 매핑.
# (band 폴더별 F가 대역마다 고유한 상수임을 이용)
FREQ_MAP = {
    8.954243: "LTE_900M",
    9.255273: "LTE_1.8G",
    9.322219: "LTE_2.1G",
    9.544068: "5G_3.5G",
    9.852785: "6G_7.125G",
    10.170262: "6G_14.8G",
}
# 그래프 x축 정렬 순서 (실제 주파수 오름차순)
BAND_ORDER = ["LTE_900M", "LTE_1.8G", "LTE_2.1G", "5G_3.5G", "6G_7.125G", "6G_14.8G"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_freq_band(df: pd.DataFrame) -> pd.DataFrame:
    """freq_band 컬럼이 없으면 F 값으로부터 복원한다."""
    if "freq_band" in df.columns:
        return df
    def _match(f_val):
        for ref, band in FREQ_MAP.items():
            if abs(f_val - ref) < 1e-3:
                return band
        return None
    df = df.copy()
    df["freq_band"] = df["F"].apply(_match)
    unmatched = df["freq_band"].isna().sum()
    if unmatched:
        print(f"[경고] F 값을 대역으로 매핑하지 못한 행 {unmatched}개 (FREQ_MAP 확인 필요)")
        df = df.dropna(subset=["freq_band"])
    return df


# ════════════════════════════════════════════════════════════════════
# 모델 정의 (main.py의 MyMLP와 동일)
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


FEATURE_COLS = ["R", "D", "H", "F"]
TARGET_COL = "PL"


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    gen = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen)


def train_one_band(band_df: pd.DataFrame, epochs: int = NUM_EPOCHS, verbose: bool = False):
    """단일 (지역, 주파수 대역) 데이터에 대해 MLP를 처음부터 학습하고 test MSE/RMSE/MAE를 반환."""
    set_seed(SEED)

    trainval, test = train_test_split(band_df, test_size=TEST_SIZE, shuffle=True, random_state=SEED)
    train, val = train_test_split(trainval, test_size=VAL_SIZE, shuffle=True, random_state=SEED)

    scaler = RobustScaler().fit(train[FEATURE_COLS].values)
    Xtr, ytr = scaler.transform(train[FEATURE_COLS].values), train[TARGET_COL].values
    Xva, yva = scaler.transform(val[FEATURE_COLS].values), val[TARGET_COL].values
    Xte, yte = scaler.transform(test[FEATURE_COLS].values), test[TARGET_COL].values

    train_loader = make_loader(Xtr, ytr, shuffle=True)
    val_loader   = make_loader(Xva, yva, shuffle=False)

    model = MyMLP(input_dim=4, hidden_dim=HIDDEN_DIM, output_dim=1, layer_num=LAYER_NUM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_losses = [criterion(model(xb.to(device)), yb.to(device)).item()
                          for xb, yb in val_loader]
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 20 == 0 or ep == epochs):
            print(f"    epoch {ep:>3}/{epochs}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32).to(device)).cpu().numpy().ravel()
    mse = float(np.mean((pred - yte) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - yte)))
    return {"mse": mse, "rmse": rmse, "mae": mae, "n_test": len(yte)}


def evaluate_region(csv_path: str, epochs: int = NUM_EPOCHS, verbose: bool = False) -> pd.DataFrame:
    region = Path(csv_path).stem
    df = pd.read_csv(csv_path, index_col=0)
    df = label_freq_band(df)

    rows = []
    for band in BAND_ORDER:
        band_df = df[df["freq_band"] == band]
        if band_df.empty:
            continue
        print(f"[{region}] {band} 학습 중... ({len(band_df)} rows)")
        metrics = train_one_band(band_df, epochs=epochs, verbose=verbose)
        rows.append({"region": region, "freq_band": band, **metrics})
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel in zip(axes, ["mse", "rmse"], ["MSE", "RMSE"]):
        for region, sub in results.groupby("region"):
            sub = sub.set_index("freq_band").reindex(BAND_ORDER).dropna()
            ax.plot(sub.index, sub[metric], marker="o", label=region)
        ax.set_xlabel("Freq Band")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Frequency Band-wise Test {ylabel}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)
        if results["region"].nunique() > 1:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n저장 완료 -> {out_path}")


def main():
    '''
    /Users/howard/Python_workspace/continual_learning_comm/.venv/bin/python simulation/freq_perform_val.py 
    --data SBS_Seoul MBC_CheongJu MBC_ChungJu MBC_DaeJeon JeonJu_Radio KBS_SunCheon TBC_DaeGu KBS_KangNeung
    '''
    ap = argparse.ArgumentParser(description="지역별 주파수 대역 성능 학습/시각화")
    ap.add_argument("--data", type=str, nargs="+", default=None,
                     help="학습할 지역 CSV 파일명(들). 확장자/경로 생략 가능. "
                          "--data-root 기준으로 찾는다 (예: --data SBS_Seoul KBS_DaeJeon MBC_ChungJu)")
    ap.add_argument("--data-root", type=str, default="MLP/DATA",
                     help="지역별 CSV가 저장된 기본 폴더 (기본값: MLP/DATA)")
    ap.add_argument("--data-dir", type=str, default=None,
                     help="지정 시, 이 폴더의 모든 CSV를 사용 (--data보다 우선)")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--out-png", type=str, default="freq_band_performance.png")
    ap.add_argument("--out-csv", type=str, default="freq_band_performance.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
 
    def resolve(name: str) -> str:
        """파일명만 주어져도 --data-root 기준으로 찾아준다. 확장자 생략도 허용."""
        candidates = [name]
        if not name.lower().endswith(".csv"):
            candidates.append(name + ".csv")
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
            joined = os.path.join(args.data_root, cand)
            if os.path.isfile(joined):
                return joined
        raise FileNotFoundError(
            f"'{name}' 을(를) 현재 경로 또는 '{args.data_root}' 에서 찾지 못했습니다."
        )
 
    if args.data_dir:
        csv_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    elif args.data:
        csv_paths = [resolve(name) for name in args.data]
    else:
        csv_paths = sorted(glob.glob(os.path.join(args.data_root, "*.csv")))
 
    if not csv_paths:
        raise SystemExit(
            "학습할 CSV를 찾지 못했습니다. --data (파일명 여러 개 가능), "
            "--data-dir(폴더), 또는 --data-root(기본 폴더)를 지정하세요."
        )
 
    print(f"학습 대상 파일 {len(csv_paths)}개:")
    for p in csv_paths:
        print(f"  - {p}")
    print()
 

    all_results = []
    for path in csv_paths:
        all_results.append(evaluate_region(path, epochs=args.epochs, verbose=args.verbose))
    results = pd.concat(all_results, ignore_index=True)

    results.to_csv(args.out_csv, index=False)
    print(f"\n결과 테이블 저장 -> {args.out_csv}")
    print(results.to_string(index=False))

    plot_results(results, args.out_png)


if __name__ == "__main__":
    main()