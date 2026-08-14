"""
─────────────────────────────────────────────────────────────────────────
new_main.py 를 라이브러리처럼 재사용하여 아래 네 가지 실험을 자동으로
수행하고, 결과를 CSV와 그래프(PNG)로 저장하는 종합 평가 스크립트입니다.

    실험 1) ER 사용 여부 비교          (use_replay = True / False)
    실험 2) 태스크 개수 민감도         (task 수를 2, 3, 4 ... 로 늘려가며 비교)
    실험 3) 하이퍼파라미터 스윕        (memory_capacity, LAMBDA)
    실험 4) 샘플링 전략 비교           (Reservoir vs LARS)

실행 방법
─────────
    이 파일을 new_main.py 와 "같은 폴더"에 두고 실행합니다.

        python evaluate_pipeline.py

    new_main.py 안의 model, train_on_task, evaluate, data_split,
    generate_csv_tasks 등을 그대로 재사용하며, 실험마다 모델 가중치와
    리플레이 버퍼를 초기화한 뒤 독립적으로 학습을 진행합니다.

주의
────
    new_main.py 최상단의 CSV 생성/학습 코드는 `if __name__ == "__main__":`
    가드 안에 있으므로, import 시점에는 모델 정의와 함수 선언만 실행되고
    실제 학습은 수행되지 않습니다.
"""

import os
import copy
import logging
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# ── new_main.py 를 모듈로 임포트 ───────────────────────────────────
#    (같은 디렉토리에 new_main.py 가 있어야 합니다)
import simulation.main as nm


# ════════════════════════════════════════════════════════════════════
# 한글 폰트 설정 (OS별 자동 탐지)
# ════════════════════════════════════════════════════════════════════
def setup_korean_font():
    """
    OS에 설치된 한글 폰트를 자동으로 탐지하여 matplotlib에 설정합니다.
    설치된 한글 폰트가 없으면 라벨을 영문으로 대체할 수 있도록
    USE_KOREAN_LABELS 플래그를 False로 반환합니다.
    """
    system = platform.system()

    candidates = {
        "Windows": ["Malgun Gothic", "맑은 고딕"],
        "Darwin":  ["AppleGothic"],
        "Linux":   ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
                   "UnDotum", "Baekmuk Dotum"],
    }.get(system, [])

    installed = {f.name for f in fm.fontManager.ttflist}

    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지
            logger.info(f"한글 폰트 설정 완료: {name}")
            return True

    # 후보 폰트가 하나도 없을 경우: 나눔고딕 설치 여부 재확인 후 실패 처리
    logger.warning(
        "설치된 한글 폰트를 찾지 못했습니다. 그래프 라벨이 깨질 수 있습니다.\n"
        "  Windows : 기본 제공 (Malgun Gothic) — 정상적으로 잡히지 않으면 재확인 필요\n"
        "  Mac     : 기본 제공 (AppleGothic)\n"
        "  Linux   : sudo apt-get install -y fonts-nanum  후 재실행\n"
        "            (설치 후 아래 캐시 삭제 필요)\n"
        "            rm -rf ~/.cache/matplotlib && python evaluate_pipeline.py"
    )
    plt.rcParams["axes.unicode_minus"] = False
    return False


# ════════════════════════════════════════════════════════════════════
# 출력 경로 / 로거
# ════════════════════════════════════════════════════════════════════
EVAL_DIR      = os.path.join(nm.save_model_dir, "evaluation")
PLOT_DIR      = os.path.join(EVAL_DIR, "plots")
RESULT_CSV_DIR = os.path.join(EVAL_DIR, "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_CSV_DIR, exist_ok=True)

_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

logger = logging.getLogger("ER_Evaluation")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(
        os.path.join(EVAL_DIR, f"evaluation_{_stamp}.log"), encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)-5s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)-5s] %(message)s"))
# ── 한글 폰트 설정 실행 (로거 정의 이후에 호출) ─────────────────────
USE_KOREAN_LABELS = setup_korean_font()


def L(korean: str, english: str) -> str:
    """
    한글 폰트가 없는 환경에서는 자동으로 영문 라벨을 사용합니다.
    (그래프 텍스트가 네모(□)로 깨지는 것을 방지)
    """
    return korean if USE_KOREAN_LABELS else english


# ════════════════════════════════════════════════════════════════════
# 공통 유틸: 모델 / 버퍼 리셋
# ════════════════════════════════════════════════════════════════════
def init_weights(m):
    """ReLU 기반 MLP에 적합한 Kaiming 초기화."""
    if isinstance(m, nm.nn.Linear):
        nm.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nm.nn.init.zeros_(m.bias)


def reset_model():
    """모델 가중치를 초기 상태로 되돌립니다."""
    nm.model.apply(init_weights)


def reset_buffer():
    """리플레이 버퍼와 seen_examples 카운터를 초기화합니다."""
    nm.memory_x       = []
    nm.memory_y       = []
    nm.memory_teacher = []
    nm.memory_loss    = []
    nm.seen_examples   = 0


def configure(use_replay: bool, use_lars: bool = False,
             lambda_val: float = 1.0, memory_capacity: int = 10000,
             num_epochs: int = None):
    """new_main 모듈의 하이퍼파라미터를 실험 조건에 맞게 설정합니다."""
    nm.use_replay       = use_replay
    nm.use_lars          = use_lars
    nm.LAMBDA            = lambda_val
    nm.memory_capacity   = memory_capacity
    if num_epochs is not None:
        nm.NUM_EPOCHS = num_epochs


# ════════════════════════════════════════════════════════════════════
# 핵심 실행기: 태스크 순서대로 학습하며 매 단계마다 전체 평가
# ════════════════════════════════════════════════════════════════════
def run_sequential_experiment(
    tasks: List[Tuple[str, pd.DataFrame]],
    label: str,
    use_replay: bool,
    use_lars: bool = False,
    lambda_val: float = 1.0,
    memory_capacity: int = 10000,
    num_epochs: int = None,
    silent_train_log: bool = True,
) -> pd.DataFrame:
    """
    tasks 를 순서대로 학습하며, 각 태스크 학습 직후
    '지금까지 학습한 모든 태스크'를 재평가하여 기록합니다.

    반환되는 DataFrame 형태 (long format):
        stage | stage_task | eval_task | mse | rmse | mae
        0     | task1.csv  | task1.csv | ..  | ..   | ..
        1     | task2.csv  | task1.csv | ..  | ..   | ..
        1     | task2.csv  | task2.csv | ..  | ..   | ..
        ...
    stage 는 0부터 시작하는 학습 순번입니다.
    """
    configure(use_replay=use_replay, use_lars=use_lars,
             lambda_val=lambda_val, memory_capacity=memory_capacity,
             num_epochs=num_epochs)
    reset_model()
    reset_buffer()

    logger.info(f"[{label}] 실험 시작 | use_replay={use_replay} | "
               f"use_lars={use_lars} | lambda={lambda_val} | "
               f"mem_cap={memory_capacity} | tasks={len(tasks)}")

    records = []

    # 원래 print 를 잠깐 무음 처리(학습 로그가 너무 길어지는 것 방지)
    _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print

    def _silent_print(*args, **kwargs):
        pass

    for stage, (file_name, df) in enumerate(tasks):
        train_data, _ = nm.data_split(df)

        if silent_train_log:
            if isinstance(__builtins__, dict):
                __builtins__["print"] = _silent_print
            else:
                __builtins__.print = _silent_print
        try:
            nm.train_on_task(train_data, task_name=f"[{label}] {file_name}")
        finally:
            if isinstance(__builtins__, dict):
                __builtins__["print"] = _orig_print
            else:
                __builtins__.print = _orig_print

        # 지금까지 학습한 모든 태스크 재평가
        for eval_idx, (eval_name, eval_df) in enumerate(tasks[: stage + 1]):
            _, val_data = nm.data_split(eval_df)

            if isinstance(__builtins__, dict):
                __builtins__["print"] = _silent_print
            else:
                __builtins__.print = _silent_print
            try:
                res = nm.evaluate(f"[{label}] {eval_name}", val_data)
            finally:
                if isinstance(__builtins__, dict):
                    __builtins__["print"] = _orig_print
                else:
                    __builtins__.print = _orig_print

            records.append({
                "label":      label,
                "stage":      stage,
                "stage_task": file_name,
                "eval_task":  eval_name,
                "mse":        res["mse"],
                "rmse":       res["rmse"],
                "mae":        res["mae"],
            })

        logger.info(f"[{label}] stage {stage} ({file_name}) 학습 및 평가 완료")

    return pd.DataFrame.from_records(records)


# ════════════════════════════════════════════════════════════════════
# 지표 계산: Average MSE / Backward Transfer(BWT)
# ════════════════════════════════════════════════════════════════════
def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    df: run_sequential_experiment() 의 반환값 (단일 label 기준)

    반환:
        avg_final_mse : 전체 학습 완료 후 모든 태스크에 대한 평균 MSE
        bwt           : Backward Transfer
                         (양수에 가까울수록 좋음, 음수가 클수록 망각 심함)
    """
    last_stage = df["stage"].max()

    # 전체 학습 완료 후 각 태스크 성능
    final_row = df[df["stage"] == last_stage]
    avg_final_mse = final_row["mse"].mean()

    # BWT 계산: 태스크 i를 "막 학습했을 때" 성능 vs "전체 학습 후" 성능
    bwt_terms = []
    tasks_learned = df["stage_task"].drop_duplicates().tolist()

    for i, task_name in enumerate(tasks_learned[:-1]):
        # 태스크 i를 막 학습한 직후(stage=i) 자기 자신 평가값
        just_after = df[(df["stage"] == i) & (df["eval_task"] == task_name)]["mse"]
        # 전체 학습 완료 후 태스크 i 평가값
        final_val  = df[(df["stage"] == last_stage) & (df["eval_task"] == task_name)]["mse"]

        if len(just_after) and len(final_val):
            # MSE는 낮을수록 좋으므로 부호를 반전하여
            # "양수 = 성능 향상, 음수 = 성능 저하(망각)" 로 해석
            bwt_terms.append(just_after.values[0] - final_val.values[0])

    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0

    return {"avg_final_mse": avg_final_mse, "bwt": bwt}


# ════════════════════════════════════════════════════════════════════
# 실험 1) ER 사용 여부 비교
# ════════════════════════════════════════════════════════════════════
def experiment_er_vs_naive(tasks, num_epochs=None) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("실험 1) ER vs Naive Fine-tuning 비교")
    logger.info("=" * 60)

    er_df    = run_sequential_experiment(
        tasks, label="ER",    use_replay=True,  use_lars=nm.use_lars,
        lambda_val=1.0, memory_capacity=10000, num_epochs=num_epochs)
    naive_df = run_sequential_experiment(
        tasks, label="Naive", use_replay=False, num_epochs=num_epochs)

    combined = pd.concat([er_df, naive_df], ignore_index=True)
    combined.to_csv(
        os.path.join(RESULT_CSV_DIR, f"exp1_er_vs_naive_{_stamp}.csv"),
        index=False, encoding="utf-8-sig")

    er_metrics    = compute_metrics(er_df)
    naive_metrics = compute_metrics(naive_df)

    logger.info(f"[ER]    avg_final_mse={er_metrics['avg_final_mse']:.4f} | "
               f"BWT={er_metrics['bwt']:.4f}")
    logger.info(f"[Naive] avg_final_mse={naive_metrics['avg_final_mse']:.4f} | "
               f"BWT={naive_metrics['bwt']:.4f}")

    return combined


# ════════════════════════════════════════════════════════════════════
# 실험 2) 태스크 개수 민감도
# ════════════════════════════════════════════════════════════════════
def experiment_task_count_sensitivity(tasks, num_epochs=None) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("실험 2) 태스크 개수 민감도 분석")
    logger.info("=" * 60)

    if len(tasks) < 3:
        logger.warning("태스크가 3개 미만이므로 태스크 수 민감도 실험은 "
                       "의미 있는 결과를 내기 어렵습니다. task_configs에 "
                       "태스크를 더 추가하는 것을 권장합니다.")

    records = []
    for k in range(2, len(tasks) + 1):
        subset = tasks[:k]

        er_df    = run_sequential_experiment(
            subset, label=f"ER_{k}tasks",    use_replay=True,
            use_lars=nm.use_lars, memory_capacity=10000, num_epochs=num_epochs)
        naive_df = run_sequential_experiment(
            subset, label=f"Naive_{k}tasks", use_replay=False,
            num_epochs=num_epochs)

        er_m    = compute_metrics(er_df)
        naive_m = compute_metrics(naive_df)

        records.append({"num_tasks": k, "method": "ER",
                        "avg_final_mse": er_m["avg_final_mse"], "bwt": er_m["bwt"]})
        records.append({"num_tasks": k, "method": "Naive",
                        "avg_final_mse": naive_m["avg_final_mse"], "bwt": naive_m["bwt"]})

        logger.info(f"[{k} tasks] ER avg_mse={er_m['avg_final_mse']:.4f} | "
                   f"Naive avg_mse={naive_m['avg_final_mse']:.4f}")

    df = pd.DataFrame.from_records(records)
    df.to_csv(
        os.path.join(RESULT_CSV_DIR, f"exp2_task_count_{_stamp}.csv"),
        index=False, encoding="utf-8-sig")
    return df


# ════════════════════════════════════════════════════════════════════
# 실험 3) 하이퍼파라미터 스윕 (memory_capacity, LAMBDA)
# ════════════════════════════════════════════════════════════════════
def experiment_hyperparameter_sweep(
    tasks,
    memory_capacities: List[int] = (500, 1000, 5000, 10000),
    lambda_values: List[float]   = (0.1, 0.5, 1.0, 2.0),
    num_epochs=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("실험 3) 하이퍼파라미터 스윕 (memory_capacity, LAMBDA)")
    logger.info("=" * 60)

    # ── 3-1) memory_capacity 스윕 (LAMBDA 고정) ────────────────────
    mem_records = []
    for cap in memory_capacities:
        df = run_sequential_experiment(
            tasks, label=f"ER_mem{cap}", use_replay=True, use_lars=nm.use_lars,
            lambda_val=1.0, memory_capacity=cap, num_epochs=num_epochs)
        m = compute_metrics(df)
        mem_records.append({"memory_capacity": cap,
                            "avg_final_mse": m["avg_final_mse"], "bwt": m["bwt"]})
        logger.info(f"[mem_capacity={cap}] avg_mse={m['avg_final_mse']:.4f} | "
                   f"BWT={m['bwt']:.4f}")

    mem_df = pd.DataFrame.from_records(mem_records)
    mem_df.to_csv(
        os.path.join(RESULT_CSV_DIR, f"exp3_memory_capacity_{_stamp}.csv"),
        index=False, encoding="utf-8-sig")

    # ── 3-2) LAMBDA 스윕 (memory_capacity 고정) ─────────────────────
    lam_records = []
    for lam in lambda_values:
        df = run_sequential_experiment(
            tasks, label=f"ER_lambda{lam}", use_replay=True, use_lars=nm.use_lars,
            lambda_val=lam, memory_capacity=10000, num_epochs=num_epochs)
        m = compute_metrics(df)
        lam_records.append({"lambda": lam,
                            "avg_final_mse": m["avg_final_mse"], "bwt": m["bwt"]})
        logger.info(f"[lambda={lam}] avg_mse={m['avg_final_mse']:.4f} | "
                   f"BWT={m['bwt']:.4f}")

    lam_df = pd.DataFrame.from_records(lam_records)
    lam_df.to_csv(
        os.path.join(RESULT_CSV_DIR, f"exp3_lambda_{_stamp}.csv"),
        index=False, encoding="utf-8-sig")

    return mem_df, lam_df


# ════════════════════════════════════════════════════════════════════
# 실험 4) 샘플링 전략 비교 (Reservoir vs LARS)
# ════════════════════════════════════════════════════════════════════
def experiment_sampling_strategy(tasks, num_epochs=None) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("실험 4) 샘플링 전략 비교 (Reservoir vs LARS)")
    logger.info("=" * 60)

    reservoir_df = run_sequential_experiment(
        tasks, label="Reservoir", use_replay=True, use_lars=False,
        lambda_val=1.0, memory_capacity=10000, num_epochs=num_epochs)
    lars_df = run_sequential_experiment(
        tasks, label="LARS", use_replay=True, use_lars=True,
        lambda_val=1.0, memory_capacity=10000, num_epochs=num_epochs)

    combined = pd.concat([reservoir_df, lars_df], ignore_index=True)
    combined.to_csv(
        os.path.join(RESULT_CSV_DIR, f"exp4_sampling_{_stamp}.csv"),
        index=False, encoding="utf-8-sig")

    res_m  = compute_metrics(reservoir_df)
    lars_m = compute_metrics(lars_df)
    logger.info(f"[Reservoir] avg_mse={res_m['avg_final_mse']:.4f} | BWT={res_m['bwt']:.4f}")
    logger.info(f"[LARS]      avg_mse={lars_m['avg_final_mse']:.4f} | BWT={lars_m['bwt']:.4f}")

    return combined


# ════════════════════════════════════════════════════════════════════
# 시각화
# ════════════════════════════════════════════════════════════════════
def plot_er_vs_naive(df: pd.DataFrame):
    """
    실험 1 결과 시각화
      (a) 태스크별 MSE 추이 (ER vs Naive, 태스크별 라인)
      (b) 최종 평균 MSE / BWT 막대 비교
    """
    tasks_order = df[df["label"] == "ER"]["stage_task"].drop_duplicates().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 태스크별 MSE 추이
    ax = axes[0]
    for eval_task in tasks_order:
        er_sub    = df[(df["label"] == "ER")    & (df["eval_task"] == eval_task)]
        naive_sub = df[(df["label"] == "Naive") & (df["eval_task"] == eval_task)]
        ax.plot(er_sub["stage"], er_sub["mse"], marker='o',
               label=f"ER - {eval_task}")
        ax.plot(naive_sub["stage"], naive_sub["mse"], marker='x', linestyle='--',
               label=f"Naive - {eval_task}")
    ax.set_xlabel(L("학습 진행 단계 (Stage)", "Training Stage"))
    ax.set_ylabel("MSE")
    ax.set_title(L("태스크별 MSE 변화: ER vs Naive", "Per-task MSE: ER vs Naive"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 최종 평균 MSE / BWT
    er_m    = compute_metrics(df[df["label"] == "ER"])
    naive_m = compute_metrics(df[df["label"] == "Naive"])

    ax2 = axes[1]
    x = [L("최종 평균 MSE", "Avg Final MSE"), "BWT"]
    er_vals    = [er_m["avg_final_mse"], er_m["bwt"]]
    naive_vals = [naive_m["avg_final_mse"], naive_m["bwt"]]

    width = 0.35
    idx = range(len(x))
    ax2.bar([i - width/2 for i in idx], er_vals, width, label="ER", color="steelblue")
    ax2.bar([i + width/2 for i in idx], naive_vals, width, label="Naive", color="orangered")
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels(x)
    ax2.set_title(L("최종 성능 비교 (MSE는 낮을수록, BWT는 0에 가까울수록 좋음)",
                    "Final Performance (lower MSE / BWT near 0 is better)"))
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"exp1_er_vs_naive_{_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"그래프 저장 완료 → {path}")


def plot_task_count_sensitivity(df: pd.DataFrame):
    """실험 2 결과 시각화: 태스크 수 증가에 따른 avg_final_mse, BWT"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method, color in [("ER", "steelblue"), ("Naive", "orangered")]:
        sub = df[df["method"] == method]
        axes[0].plot(sub["num_tasks"], sub["avg_final_mse"],
                    marker='o', label=method, color=color)
        axes[1].plot(sub["num_tasks"], sub["bwt"],
                    marker='o', label=method, color=color)

    axes[0].set_xlabel(L("태스크 개수", "Number of Tasks"))
    axes[0].set_ylabel(L("최종 평균 MSE", "Avg Final MSE"))
    axes[0].set_title(L("태스크 개수 증가에 따른 평균 성능", "Avg Performance vs Task Count"))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel(L("태스크 개수", "Number of Tasks"))
    axes[1].set_ylabel("BWT")
    axes[1].set_title(L("태스크 개수 증가에 따른 망각 정도(BWT)", "Forgetting(BWT) vs Task Count"))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"exp2_task_count_{_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"그래프 저장 완료 → {path}")


def plot_hyperparameter_sweep(mem_df: pd.DataFrame, lam_df: pd.DataFrame):
    """실험 3 결과 시각화: memory_capacity, LAMBDA 각각의 영향"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(mem_df["memory_capacity"], mem_df["avg_final_mse"],
                marker='o', color="seagreen")
    axes[0].set_xlabel("memory_capacity")
    axes[0].set_ylabel(L("최종 평균 MSE", "Avg Final MSE"))
    axes[0].set_title(L("버퍼 크기에 따른 성능 변화", "Performance vs Buffer Size"))
    axes[0].grid(alpha=0.3)

    axes[1].plot(lam_df["lambda"], lam_df["avg_final_mse"],
                marker='o', color="darkorange")
    axes[1].set_xlabel(L("LAMBDA (replay loss 가중치)", "LAMBDA (replay loss weight)"))
    axes[1].set_ylabel(L("최종 평균 MSE", "Avg Final MSE"))
    axes[1].set_title(L("LAMBDA 값에 따른 성능 변화", "Performance vs LAMBDA"))
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"exp3_hyperparam_sweep_{_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"그래프 저장 완료 → {path}")


def plot_sampling_strategy(df: pd.DataFrame):
    """실험 4 결과 시각화: Reservoir vs LARS"""
    res_m  = compute_metrics(df[df["label"] == "Reservoir"])
    lars_m = compute_metrics(df[df["label"] == "LARS"])

    fig, ax = plt.subplots(figsize=(6, 5))
    x = ["Avg Final MSE", "BWT"]
    res_vals  = [res_m["avg_final_mse"], res_m["bwt"]]
    lars_vals = [lars_m["avg_final_mse"], lars_m["bwt"]]

    width = 0.35
    idx = range(len(x))
    ax.bar([i - width/2 for i in idx], res_vals, width, label="Reservoir", color="slateblue")
    ax.bar([i + width/2 for i in idx], lars_vals, width, label="LARS", color="crimson")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(x)
    ax.set_title("샘플링 전략 비교: Reservoir vs LARS")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"exp4_sampling_{_stamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"그래프 저장 완료 → {path}")


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    logger.info("종합 평가 파이프라인 시작")

    # ── 실험용 epoch 수 (전체 실험 반복이 많으므로 기본보다 축소 권장) ──
    EXPERIMENT_EPOCHS = min(nm.NUM_EPOCHS, 15)

    # ── 0. 데이터 준비 (new_main.task_configs 재사용) ────────────────
    tasks = nm.generate_csv_tasks(nm.task_configs)
    logger.info(f"총 {len(tasks)}개 태스크 로드: {[t[0] for t in tasks]}")

    if len(tasks) < 2:
        raise RuntimeError(
            "실험을 위해서는 최소 2개 이상의 태스크(csv)가 필요합니다. "
            "new_main.py 의 task_configs 에 태스크를 추가하세요."
        )

    # ── 1. ER vs Naive ───────────────────────────────────────────────
    df1 = experiment_er_vs_naive(tasks, num_epochs=EXPERIMENT_EPOCHS)
    plot_er_vs_naive(df1)

    # ── 2. 태스크 개수 민감도 ─────────────────────────────────────────
    df2 = experiment_task_count_sensitivity(tasks, num_epochs=EXPERIMENT_EPOCHS)
    plot_task_count_sensitivity(df2)

    # ── 3. 하이퍼파라미터 스윕 ────────────────────────────────────────
    mem_df, lam_df = experiment_hyperparameter_sweep(
        tasks,
        memory_capacities=(500, 1000, 5000, 10000),
        lambda_values=(0.1, 0.5, 1.0, 2.0),
        num_epochs=EXPERIMENT_EPOCHS,
    )
    plot_hyperparameter_sweep(mem_df, lam_df)

    # ── 4. 샘플링 전략 비교 ──────────────────────────────────────────
    df4 = experiment_sampling_strategy(tasks, num_epochs=EXPERIMENT_EPOCHS)
    plot_sampling_strategy(df4)

    logger.info("모든 실험 완료")
    logger.info(f"결과 CSV 경로 → {RESULT_CSV_DIR}")
    logger.info(f"그래프 경로   → {PLOT_DIR}")