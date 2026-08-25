"""
pathloss_formula_comparison.py
─────────────────────────────────────────────────────────────────────────
[우선순위 3] 전통적 경로손실 공식(Log-distance path loss model)과의 비교

머신러닝을 전혀 쓰지 않는 가장 단순한 baseline인 log-distance 모델을
main.py와 동일한 데이터/분할로 fitting·평가하여, "왜 단순 공식 대신
AI 모델(MLP+ER)을 쓰는가"에 대한 정량적 근거를 만듭니다.

Log-distance path loss model:
    PL(d) = PL0 + 10 * n * log10(d / d0) + X
    - PL0 : 기준 거리 d0에서의 경로손실 (절편)
    - n   : 경로손실 지수 (거리에 따른 감쇠율. 자유공간 2, 시가지 3~5 등)
    - X   : 그림자 페이딩(shadow fading). 평균 0인 확률 변수로 가정하고
            점 추정(point prediction)에서는 0으로 둡니다. 대신 학습
            잔차의 표준편차를 sigma_X로 함께 보고해 "공식이 설명하지
            못하고 남긴 변동성"을 드러냅니다.

PL0, n은 태스크(=지역)별 train 데이터에 scipy.optimize.curve_fit으로
최소자승 피팅합니다(수렴하지 않으면 numpy.polyfit 선형회귀로 대체).
main.py의 data_split과 완전히 동일한 train/test 분할(SEED, TEST_SIZE
동일)을 사용하므로, 같은 test set에 대해 MLP+ER과 공정하게 RMSE를
비교할 수 있습니다.

사전 조건:
    cl_comparison.py를 먼저 실행해 두면 그 결과(ER의 최종 성능)와
    자동으로 비교합니다. 없어도 공식 자체의 fitting/평가는 정상
    수행되며, ER과의 비교 컬럼만 생략됩니다.

결과:
    simulation/ER_/model/pathloss_formula_comparison_{FREQ_BAND}.csv
        task, PL0, n, sigma_X, mse, rmse, mae,
        [er_mse, er_rmse, er_mae, rmse_gap, rmse_ratio]
    simulation/ER_/model/evaluation_plots/pathloss_formula_comparison_{FREQ_BAND}.png

실행 방법:
    python pathloss_formula_comparison.py
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import main as m


D0 = 1.0  # 기준 거리(R 컬럼과 같은 단위). d=0에서 log가 발산하므로
         # d0=1로 두고, d<1e-3인 샘플은 아래에서 clip합니다.


def _log_distance(d: np.ndarray, pl0: float, n: float) -> np.ndarray:
    d_safe = np.clip(d, 1e-3, None)
    return pl0 + 10 * n * np.log10(d_safe / D0)


def fit_and_eval_task(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """
    한 태스크(지역)의 train 데이터로 PL0, n을 curve_fit으로 피팅하고,
    같은 태스크의 test 데이터로 MSE/RMSE/MAE를 계산합니다. main.py의
    evaluate()가 쓰는 정의(제곱오차 평균 → RMSE, 절대오차 평균 → MAE)와
    동일한 방식으로 지표를 계산해 MLP+ER과 직접 비교 가능하게 합니다.
    """
    d_train = train_df["R"].values.astype("float64")
    pl_train = train_df[m.target].values.astype("float64")

    # 초기값: PL0=train PL의 중앙값 근처, n=3(자유공간~시가지 사이 일반값)
    p0 = [float(np.median(pl_train)), 3.0]
    try:
        (pl0, n), _ = curve_fit(_log_distance, d_train, pl_train, p0=p0, maxfev=10000)
    except RuntimeError:
        # curve_fit이 수렴하지 않으면 log10(d)에 대한 선형회귀로 대체합니다.
        # PL = a + b*log10(d) 형태이므로 n = b/10, PL0 = a.
        log_d = np.log10(np.clip(d_train, 1e-3, None) / D0)
        b, a = np.polyfit(log_d, pl_train, 1)
        pl0, n = float(a), float(b) / 10.0

    d_test = test_df["R"].values.astype("float64")
    pl_test = test_df[m.target].values.astype("float64")
    pred = _log_distance(d_test, pl0, n)

    resid_train = pl_train - _log_distance(d_train, pl0, n)
    sigma_x = float(np.std(resid_train))

    mse = float(np.mean((pred - pl_test) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - pl_test)))

    return {"PL0": float(pl0), "n": float(n), "sigma_X": sigma_x,
           "mse": mse, "rmse": rmse, "mae": mae}


def run() -> pd.DataFrame:
    print(f"=== Log-distance 공식 baseline ({m.FREQ_BAND}) ===")
    # main.py와 완전히 동일한 CSV 생성/train-test 분할 (공정 비교의 전제)
    tasks = m.generate_csv_tasks(m.task_configs)

    rows: List[Dict] = []
    for file_name, df in tasks:
        train_df, test_df = m.data_split(df)
        result = fit_and_eval_task(train_df, test_df)
        result["task"] = file_name
        rows.append(result)
        print(f"  {file_name}: PL0={result['PL0']:.2f}, n={result['n']:.3f}, "
             f"RMSE={result['rmse']:.4f}")

    out_df = pd.DataFrame(rows)[["task", "PL0", "n", "sigma_X", "mse", "rmse", "mae"]]

    # ── ER 결과와 병합(cl_comparison.py를 먼저 실행했다면) ────────────
    cl_csv = os.path.join(m.save_model_dir, f"cl_comparison_{m.FREQ_BAND}.csv")
    if os.path.exists(cl_csv):
        cl_df = pd.read_csv(cl_csv)
        er_df = cl_df[cl_df["method"] == "ER"]
        if len(er_df):
            last_stage = er_df["stage"].max()
            er_final = (er_df[er_df["stage"] == last_stage]
                       .rename(columns={"eval_task": "task", "mse": "er_mse",
                                        "rmse": "er_rmse", "mae": "er_mae"})
                       [["task", "er_mse", "er_rmse", "er_mae"]])
            out_df = out_df.merge(er_final, on="task", how="left")
            # 공식이 ER보다 얼마나 나쁜지: 절대 차이와 배수 둘 다 제공
            out_df["rmse_gap"] = out_df["rmse"] - out_df["er_rmse"]
            out_df["rmse_ratio"] = out_df["rmse"] / out_df["er_rmse"]
    else:
        print(f"[안내] {cl_csv} 가 없어 ER과의 비교는 건너뜁니다. "
             "cl_comparison.py를 먼저 실행하면 ER 대비 비교 컬럼까지 함께 생성됩니다.")

    out_path = os.path.join(m.save_model_dir, f"pathloss_formula_comparison_{m.FREQ_BAND}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[저장] {out_path}")

    if "er_rmse" in out_df.columns:
        print("\n=== 공식 vs MLP+ER RMSE 배수(공식이 ER보다 몇 배 나쁜지) ===")
        print(out_df[["task", "rmse", "er_rmse", "rmse_ratio"]].to_string(index=False))

    _plot(out_df, m.FREQ_BAND)
    return out_df


def _plot(df: pd.DataFrame, freq_band: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(m.save_model_dir, "evaluation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df["rmse"], width, label="Log-distance formula")
    if "er_rmse" in df.columns:
        ax.bar(x + width / 2, df["er_rmse"], width, label="MLP + ER")

    ax.set_xticks(x)
    ax.set_xticklabels(df["task"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title(f"Log-distance formula vs MLP+ER - {freq_band}")
    ax.legend()
    fig.tight_layout()

    save_path = os.path.join(plot_dir, f"pathloss_formula_comparison_{freq_band}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[저장] {save_path}")


if __name__ == "__main__":
    run()