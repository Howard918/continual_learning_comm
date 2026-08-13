"""
evaluation.py
─────────────────────────────────────────────────────────────────────────
new_main.py가 저장한 결과 CSV(validation_*.csv, results_*.csv)를 읽어
시각화만 전담하는 스크립트입니다. 학습 로직은 전혀 포함하지 않습니다.

전제:
    new_main.py를 여러 하이퍼파라미터 조합으로 반복 실행하여
    simulation/ER_/model/history/validation_{run_name}.csv
    simulation/ER_/model/results_{run_name}.csv
    파일들이 이미 여러 개 쌓여 있어야 합니다. (run_name은 new_main.py의
    `save` 변수 값입니다.)

포함된 그래프:
    a. validation 결과, 범례=데이터(태스크/도시), 하나의 그래프
    b. test 결과, 범례=데이터(태스크/도시), 하나의 그래프
    c. alpha/beta 스윕별 test 결과 (alpha 비교 시 beta=0, beta 비교 시 alpha=0.5 로 고정된 실행만 사용)
    d. 주파수 대역(LTE/5G/6G)별 test 결과
    e. hidden_node/layer 스윕별 test 결과 (hidden_node 비교 시 layer=2, layer 비교 시 hidden_node=64 로 고정된 실행만 사용)
"""

import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform


# ════════════════════════════════════════════════════════════════════
# 경로 설정 (new_main.py와 동일한 규칙을 따릅니다)
# ════════════════════════════════════════════════════════════════════
SAVE_MODEL_DIR = "simulation/ER_/model/"
HISTORY_DIR    = os.path.join(SAVE_MODEL_DIR, "history")
EVAL_PLOT_DIR  = os.path.join(SAVE_MODEL_DIR, "evaluation_plots")
os.makedirs(EVAL_PLOT_DIR, exist_ok=True)


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
# 데이터 로딩 유틸
# ════════════════════════════════════════════════════════════════════
def _filter_eq(df: pd.DataFrame, col: str, val) -> pd.DataFrame:
    """
    부동소수점 비교 오차(예: BETA=0 을 실행마다 0 / 0.0 등으로 다르게
    기록했을 때)로 인해 필터링에서 행이 누락되는 것을 방지합니다.
    """
    if df.empty or col not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[col]):
        return df[np.isclose(df[col].astype(float), float(val))]
    return df[df[col] == val]


def load_all_results(aggregate_over_eval_task: bool = True,
                     sweep_tag: Optional[str] = None) -> pd.DataFrame:
    """
    save_model_dir에 저장된 results_*.csv들을 하나로 합쳐서 반환합니다.
    각 행에는 run_name, alpha, beta, hidden_dim, layer_num, freq_band 등
    실행 조건이 함께 기록되어 있습니다.

    sweep_tag를 지정하면, 파일을 읽는 시점부터 그 sweep_tag에 해당하는
    run만 골라서 로드합니다. 즉 다른 목적의 스윕(alpha_sweep, freq_sweep,
    hidden_dim_sweep 등)에서 나온 파일은 애초에 합치기(concat) 대상에도
    포함되지 않습니다. 이렇게 해야 서로 다른 스윕의 run이 하이퍼파라미터
    조건이 우연히 같다는 이유만으로 섞여 들어가는 것을 원천적으로 막을 수
    있습니다. sweep_tag=None이면(기본값) 필터링 없이 전체를 로드합니다.

    results_*.csv는 (b) 항목의 목적을 위해 stage당 eval_task 개수만큼의
    행을 담고 있습니다. c/d/e처럼 "그 시점까지 학습한 모든 태스크에 대한
    평균 성능"으로 하이퍼파라미터/설정을 비교하고 싶을 때는
    aggregate_over_eval_task=True(기본값)로 두면, run_name+stage 단위로
    eval_task 평균을 낸 뒤 반환합니다.
    """
    paths = glob.glob(os.path.join(SAVE_MODEL_DIR, "results_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"{SAVE_MODEL_DIR} 에서 results_*.csv 파일을 찾지 못했습니다. "
            "new_main.py를 먼저 실행해 결과를 생성해야 합니다."
        )

    dfs = []
    for p in paths:
        d = pd.read_csv(p)

        if sweep_tag is not None:
            if "sweep_tag" not in d.columns:
                print(f"[경고] {os.path.basename(p)}에 sweep_tag 컬럼이 없어 "
                     "이 sweep 비교에서 건너뜁니다. new_main.py를 최신 "
                     "버전으로 다시 실행해야 sweep_tag로 로드할 수 있습니다.")
                continue
            d = d[d["sweep_tag"] == sweep_tag]
            if d.empty:
                continue

        dfs.append(d)

    if not dfs:
        if sweep_tag is not None:
            print(f"[경고] sweep_tag='{sweep_tag}' 에 해당하는 results_*.csv가 "
                 "없습니다.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    tag_note = f" (sweep_tag='{sweep_tag}')" if sweep_tag else ""
    print(f"[로드] results_*.csv {len(dfs)}/{len(paths)}개 파일{tag_note} 합침")
    print(f"[로드] run_name 종류: {sorted(df['run_name'].unique())}")

    if not aggregate_over_eval_task:
        return df

    meta_cols   = [c for c in df.columns
                  if c not in ("stage_task", "eval_task", "mse", "rmse", "mae")]
    agg = (df.groupby(meta_cols, as_index=False)[["mse", "rmse", "mae"]]
            .mean())
    return agg


def load_validation(run_name: str) -> pd.DataFrame:
    """지정한 run_name의 validation_*.csv(태스크·epoch별 val loss)를 불러옵니다."""
    path = os.path.join(HISTORY_DIR, f"validation_{run_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 를 찾지 못했습니다.")
    return pd.read_csv(path)


def load_results(run_name: str) -> pd.DataFrame:
    """지정한 run_name의 results_*.csv(태스크별 test 결과)를 불러옵니다."""
    path = os.path.join(SAVE_MODEL_DIR, f"results_{run_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 를 찾지 못했습니다.")
    return pd.read_csv(path)


def _save(fig, filename: str):
    path = os.path.join(EVAL_PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[그래프 저장] {path}")


# ════════════════════════════════════════════════════════════════════
# a. Validation 결과: 범례 = 데이터(태스크), 하나의 그래프
# ════════════════════════════════════════════════════════════════════
def plot_validation_by_task(run_name: str, metric: str = "val_loss"):
    """
    하나의 run(run_name) 안에서, 태스크(도시)별 epoch에 따른 validation
    loss를 하나의 그래프에 겹쳐 그립니다. 범례는 태스크(데이터) 이름입니다.
    """
    df = load_validation(run_name)

    fig, ax = plt.subplots(figsize=(10, 6))
    for task_name, sub in df.groupby("task", sort=False):
        ax.plot(sub["epoch"], sub[metric], label=task_name, linewidth=1)

    ax.set_xlabel(L("Epoch", "Epoch"))
    ax.set_ylabel(L("Validation Loss (MSE)", "Validation Loss (MSE)"))
    ax.set_title(L(f"[{run_name}] 태스크별 Validation Loss",
                   f"[{run_name}] Validation Loss by Task"))
    ax.legend(title=L("데이터", "Data"), fontsize=7, ncol=2,
             bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)

    _save(fig, f"a_validation_by_task_{run_name}.png")


# ════════════════════════════════════════════════════════════════════
# b. Test 결과: 범례 = 데이터(태스크), 하나의 그래프
# ════════════════════════════════════════════════════════════════════
def plot_test_by_task(run_name: str, metric: str = "mse"):
    """
    하나의 run(run_name) 안에서, 각 태스크(도시)의 test set이 계속
    다른 태스크들을 이어서 학습해 나가는 동안 어떻게 변화하는지를
    하나의 그래프에 겹쳐 그립니다.

    예) 1번 태스크 학습 직후 1번 test를 평가하고, 2번 태스크 학습
    직후에는 1번 test와 2번 test를 각각 따로 평가하는 식으로 기록된
    results CSV(stage, stage_task, eval_task, mse, rmse, mae)를 읽어,
    eval_task(=데이터)별로 하나의 선을 그립니다. 각 선은 그 태스크가
    처음 학습된 stage부터 시작되며, 이후 다른 태스크가 추가로 학습될
    때마다 그 시점의 성능이 이어서 찍힙니다.
    """
    df = load_results(run_name)

    fig, ax = plt.subplots(figsize=(11, 6))
    for eval_task, sub in df.groupby("eval_task", sort=False):
        sub = sub.sort_values("stage")
        ax.plot(sub["stage"], sub[metric], marker='o', markersize=3, label=eval_task)

    ax.set_xlabel(L("학습 순서 (Stage)", "Training Stage"))
    ax.set_ylabel(metric.upper())
    ax.set_title(L(f"[{run_name}] 태스크별 Test {metric.upper()} 변화 (이후 학습이 진행됨에 따라)",
                   f"[{run_name}] Test {metric.upper()} per Task as Training Progresses"))
    ax.legend(title=L("데이터", "Data"), fontsize=7, ncol=2,
             bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)

    _save(fig, f"b_test_by_task_{run_name}_{metric}.png")


# ════════════════════════════════════════════════════════════════════
# c. Alpha / Beta 스윕 비교
# ════════════════════════════════════════════════════════════════════
def plot_hyperparam_sweep(param: str, fixed: dict, metric: str = "mse",
                          sweep_tag: Optional[str] = None):
    """
    param(alpha 또는 beta)의 값을 바꿔가며 실행한 여러 run들을 비교합니다.

    sweep_tag를 지정하면(권장) 먼저 new_main.py의 SWEEP_TAG가 그 값과
    일치하는 run만 남긴 뒤 비교합니다. 이렇게 해야 다른 목적의 스윕
    (예: freq_sweep)에서 나온 run이 alpha/beta 조건이 우연히 같다는
    이유만으로 이 비교에 섞여 들어가는 것을 막을 수 있습니다.

    fixed에 지정한 다른 하이퍼파라미터(예: beta=0)는 값이 고정된 run만
    필터링해서 사용합니다. 각 run(=param 값)이 하나의 선으로 그려지며,
    범례는 param 값입니다.
    """
    df = load_all_results(sweep_tag=sweep_tag)

    for key, val in fixed.items():
        df = _filter_eq(df, key, val)

    if df.empty:
        print(f"[경고] {param} 스윕: sweep_tag={sweep_tag}, fixed={fixed} "
             f"조건에 맞는 결과가 없습니다.")
        return

    unique_vals = sorted(df[param].unique())
    print(f"[진단] {param} 스윕: {df['run_name'].nunique()}개 run, "
         f"{param} 값 종류: {unique_vals}")
    if len(unique_vals) < 2:
        print(f"[경고] {param} 값이 {len(unique_vals)}종류밖에 없어 "
             f"비교 그래프의 의미가 없습니다. new_main.py에서 {param}을 "
             f"다르게 바꿔가며(그리고 매번 다른 save 이름으로!) 더 실행해보세요.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for val, sub in df.groupby(param, sort=True):
        sub = sub.sort_values("stage")
        ax.plot(sub["stage"], sub[metric], marker='o', markersize=3,
               label=f"{param}={val}")

    ax.set_xlabel(L("학습 순서 (Stage)", "Training Stage"))
    ax.set_ylabel(metric.upper())
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
    ax.set_title(L(f"{param} 값에 따른 Test {metric.upper()} 비교 ({fixed_str} 고정)",
                   f"Test {metric.upper()} by {param} (fixed {fixed_str})"))
    ax.legend(title=param)
    ax.grid(alpha=0.3)

    _save(fig, f"c_{param}_sweep_{metric}({fixed_str}).png")


def plot_alpha_beta_sweep(metric: str = "mse"):
    """alpha 스윕과 beta 스윕을 각각 그립니다."""
    plot_hyperparam_sweep("alpha", fixed={"beta": 0.0}, metric=metric,
                          sweep_tag="alpha_sweep")
    plot_hyperparam_sweep("beta",  fixed={"alpha": 0.3}, metric=metric,
                          sweep_tag="beta_sweep")


# ════════════════════════════════════════════════════════════════════
# d. 주파수 대역(LTE/5G/6G)별 비교
# ════════════════════════════════════════════════════════════════════
def plot_freq_band_sweep(metric: str = "mse"):
    """
    freq_band(LTE_900M, LTE_1.8G, LTE_2.1G, 5G_3.5G, 6G_7.125G, 6G_14.8G)별로
    실행한 run들을 하나의 그래프에서 비교합니다. 범례는 주파수 대역입니다.

    new_main.py에서 SWEEP_TAG="freq_sweep"로 설정하고 실행한 run만
    사용합니다. (alpha/beta 스윕 등 다른 목적의 run이 주파수 조건이
    우연히 같다는 이유만으로 섞여 들어가는 것을 방지합니다.)
    """
    df = load_all_results(sweep_tag="freq_sweep")

    if df.empty:
        print("[경고] freq_sweep 태그가 붙은 결과가 없습니다.")
        return

    unique_bands = sorted(df["freq_band"].unique())
    print(f"[진단] freq_band 종류: {unique_bands}")
    if len(unique_bands) < 2:
        print("[경고] freq_band가 1종류밖에 없어 비교 그래프의 의미가 없습니다. "
             "new_main.py에서 freq를 바꿔가며(그리고 매번 다른 save 이름으로!) "
             "더 실행해보세요.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for band, sub in df.groupby("freq_band", sort=True):
        sub = sub.sort_values("stage")
        ax.plot(sub["stage"], sub[metric], marker='o', markersize=3, label=band)

    ax.set_xlabel(L("학습 순서 (Stage)", "Training Stage"))
    ax.set_ylabel(metric.upper())
    ax.set_title(L(f"주파수 대역별 Test {metric.upper()} 비교",
                   f"Test {metric.upper()} by Frequency Band"))
    ax.legend(title=L("주파수 대역", "Frequency Band"))
    ax.grid(alpha=0.3)

    _save(fig, f"d_freq_band_sweep_{metric}.png")


# ════════════════════════════════════════════════════════════════════
# e. MLP 구조(hidden_node / layer) 스윕 비교
# ════════════════════════════════════════════════════════════════════
def plot_architecture_sweep(metric: str = "mse"):
    """
    hidden_node 스윕(layer_num=2 고정)과 layer 스윕(hidden_dim=64 고정)을
    각각 그립니다.

    new_main.py에서 SWEEP_TAG="hidden_sweep" / "layer_sweep"으로 설정하고
    실행한 run만 각각 사용합니다.
    """
    # hidden_node 스윕: sweep_tag="hidden_dim_sweep" + layer_num=2 로 고정된 run만 사용
    hidden_df = load_all_results(sweep_tag="hidden_dim_sweep")
    hidden_df = _filter_eq(hidden_df, "layer_num", 2)
    if hidden_df.empty:
        print("[경고] hidden_node 스윕: sweep_tag='hidden_dim_sweep' + "
             "layer_num=2로 고정된 결과가 없습니다.")
    else:
        unique_hd = sorted(hidden_df["hidden_dim"].unique())
        print(f"[진단] hidden_node 스윕: hidden_dim 값 종류: {unique_hd}")
        if len(unique_hd) < 2:
            print("[경고] hidden_dim이 1종류밖에 없어 비교 그래프의 의미가 없습니다. "
                 "new_main.py에서 HIDDEN_DIM을 바꿔가며(그리고 매번 다른 save "
                 "이름으로!) 더 실행해보세요.")

        fig, ax = plt.subplots(figsize=(10, 6))
        for hd, sub in hidden_df.groupby("hidden_dim", sort=True):
            sub = sub.sort_values("stage")
            ax.plot(sub["stage"], sub[metric], marker='o', markersize=3,
                   label=f"hidden_node={hd}")
        ax.set_xlabel(L("학습 순서 (Stage)", "Training Stage"))
        ax.set_ylabel(metric.upper())
        ax.set_title(L(f"hidden_node 값에 따른 Test {metric.upper()} 비교 (layer=2 고정)",
                       f"Test {metric.upper()} by hidden_node (layer=2 fixed)"))
        ax.legend(title="hidden_node")
        ax.grid(alpha=0.3)
        _save(fig, f"e_hidden_dim_sweep_{metric}.png")

    # layer 스윕: sweep_tag="layer_num_sweep" + hidden_dim=64 로 고정된 run만 사용
    layer_df = load_all_results(sweep_tag="layer_num_sweep")
    layer_df = _filter_eq(layer_df, "hidden_dim", 64)
    if layer_df.empty:
        print("[경고] layer 스윕: sweep_tag='layer_num_sweep' + "
             "hidden_dim=64로 고정된 결과가 없습니다.")
    else:
        unique_ln = sorted(layer_df["layer_num"].unique())
        print(f"[진단] layer 스윕: layer_num 값 종류: {unique_ln}")
        if len(unique_ln) < 2:
            print("[경고] layer_num이 1종류밖에 없어 비교 그래프의 의미가 없습니다. "
                 "new_main.py에서 LAYER_NUM을 바꿔가며(그리고 매번 다른 save "
                 "이름으로!) 더 실행해보세요.")

        fig, ax = plt.subplots(figsize=(10, 6))
        for ln, sub in layer_df.groupby("layer_num", sort=True):
            sub = sub.sort_values("stage")
            ax.plot(sub["stage"], sub[metric], marker='o', markersize=3,
                   label=f"layer={ln}")
        ax.set_xlabel(L("학습 순서 (Stage)", "Training Stage"))
        ax.set_ylabel(metric.upper())
        ax.set_title(L(f"layer 수에 따른 Test {metric.upper()} 비교 (hidden_node=64 고정)",
                       f"Test {metric.upper()} by layer count (hidden_node=64 fixed)"))
        ax.legend(title="layer")
        ax.grid(alpha=0.3)
        _save(fig, f"e_layer_num_sweep_{metric}.png")


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # a, b: 단일 run에 대한 태스크별 breakdown.
    # new_main.py의 `save` 변수와 동일한 값을 지정해야 합니다.
    TARGET_RUN_NAME = "alpha_sweep_alpha0.5_beta0.7_hd64_layer2_LTE_2.1G"

    plot_validation_by_task(TARGET_RUN_NAME)
    plot_test_by_task(TARGET_RUN_NAME, metric="mse")
    plot_test_by_task(TARGET_RUN_NAME, metric="rmse")

    # c: alpha(beta=0 고정) / beta(alpha=0.5 고정) 스윕 비교
    #    → beta=0으로 실행한 여러 alpha run들과, alpha=0.5로 실행한
    #      여러 beta run들이 미리 저장되어 있어야 합니다.
    plot_alpha_beta_sweep(metric="mse")

    # d: 주파수 대역별 비교
    #    → freq를 LTE(900M/1.8G/2.1G), 5G(3.5G), 6G(7.125G/14.8G)로
    #      바꿔가며 실행한 결과들이 미리 저장되어 있어야 합니다.
    plot_freq_band_sweep(metric="mse")

    # e: hidden_node(layer=2 고정) / layer(hidden_node=64 고정) 스윕 비교
    #    → layer_num=2로 실행한 여러 hidden_dim run들과, hidden_dim=64로
    #      실행한 여러 layer_num run들이 미리 저장되어 있어야 합니다.
    plot_architecture_sweep(metric="mse")

    print("\n모든 평가 그래프 생성 완료.")