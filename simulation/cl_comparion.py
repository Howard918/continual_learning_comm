"""
cl_comparison.py
─────────────────────────────────────────────────────────────────────────
논문 "Path Loss Prediction Model based on Artificial Intelligence using
Experience Replay"의 실험 섹션 중 아래 두 가지를 구현합니다.

[우선순위 1] Naive Fine-tuning vs ER (핵심 실험)
    - main.py를 거의 그대로 재사용하되 USE_REPLAY만 True/False로 바꿔
      두 baseline(ER, Naive)을 만듭니다. HIDDEN_DIM/LAYER_NUM/BATCH_SIZE/
      NUM_EPOCHS/LR/ALPHA/SEED/USE_SCALING은 main.py의 값을 그대로
      사용하므로 두 방법은 "리플레이 버퍼 유무"만 다릅니다.
    - "리플레이 버퍼가 없으면 이전 태스크를 잊어버린다(catastrophic
      forgetting)"는 것과, "ER을 쓰면 forgetting이 크게 줄어든다"는
      본 논문의 핵심 주장을 Average Accuracy와 BWT로 정량적으로
      증명하기 위한 실험입니다.
    - ★ 이 실험의 목적: ER의 BWT(절댓값)가 Naive의 BWT(절댓값)보다
      0에 훨씬 가까운 것을 보여주는 것. (compute_metrics 함수 참고)

[우선순위 2] 다른 Continual Learning 기법과의 비교
    - EWC(Elastic Weight Consolidation): 파라미터 정규화 기반. 리플레이
      버퍼 없이, "이전 태스크에 중요했던(민감했던) 파라미터는 적게
      바꾼다"는 방식으로 forgetting을 억제합니다.
    - LwF(Learning without Forgetting): 지식 증류(distillation) 기반.
      리플레이 버퍼 없이, "새 태스크의 입력을 넣어도 이전 모델(teacher)의
      예측과 크게 다르지 않게 예측한다"는 방식으로 forgetting을
      억제합니다.
    ER(리플레이) vs EWC/LwF(정규화·증류, 버퍼 없음)는 서로 다른 계열의
    CL 기법이므로, 이 실험은 "이 문제(PL 예측)에서 리플레이 기반 접근이
    다른 계열의 CL 기법보다 효과적인가?"를 보여주는 근거가 됩니다.

main.py와 동일한 데이터 파이프라인(generate_csv_tasks, data_split,
features, target, SEED, TEST_SIZE, RobustScaler 관련 함수)을 그대로
import해서 사용하므로, 태스크 순서/분할/스케일링 방식이 main.py와
완전히 동일하게 유지되어 네 가지 방법이 공정하게 비교됩니다.

실행 결과:
    simulation/ER_/model/cl_comparison_{FREQ_BAND}.csv
        method, stage, stage_task, eval_task, mse, rmse, mae, freq_band
    simulation/ER_/model/cl_comparison_summary_{FREQ_BAND}.csv
        method, num_tasks, avg_accuracy_mse, bwt_mse
    simulation/ER_/model/evaluation_plots/cl_comparison_{FREQ_BAND}.png

실행 방법:
    python cl_comparison.py
"""

import os
import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

import main as m


# ════════════════════════════════════════════════════════════════════
# 공통 상태 초기화
# ════════════════════════════════════════════════════════════════════
def reset_shared_state() -> None:
    """
    main.py가 전역(global)으로 들고 있는 학습 상태를 모두 초기화합니다.
    ER/Naive/EWC/LwF 네 가지 방법을 같은 프로세스 안에서 순서대로
    실행하기 때문에, 이전 방법이 채워놓은 리플레이 버퍼/검증 버퍼/
    스케일러/history가 다음 방법으로 새어 들어가지 않도록 반드시
    각 방법 실행 직전에 호출해야 합니다.
    """
    m.memory_x, m.memory_y, m.memory_teacher, m.memory_loss = [], [], [], []
    m.seen_examples = 0
    m.val_memory_x, m.val_memory_y = [], []
    m.val_seen_examples = 0
    m.CURRENT_SCALER = None
    m.history_all = {}


def fresh_model():
    """
    ER/Naive/EWC/LwF 각 방법이 서로 다른 초기 가중치에서 시작하지
    않도록(=공정한 비교) 매 방법마다 완전히 새로운 모델을 만듭니다.
    HIDDEN_DIM/LAYER_NUM은 main.py의 값을 그대로 사용합니다.
    """
    return m.MyMLP(input_dim=4, hidden_dim=m.HIDDEN_DIM,
                    output_dim=1, layer_num=m.LAYER_NUM).to(m.device)


# ════════════════════════════════════════════════════════════════════
# EWC / LwF 전용: 리플레이 버퍼를 쓰지 않는 순수 학습 루프
# ════════════════════════════════════════════════════════════════════
# main.py의 train_epoch/build_dataloader는 리플레이 버퍼 혼합 배치
# (current+replay, is_replay 플래그)를 전제로 짜여 있습니다. EWC/LwF는
# "리플레이 버퍼 없이 정규화/증류만으로 forgetting을 억제"하는 다른
# 계열의 기법이므로, 여기서는 순수 (X, y) 배치만 쓰는 간단한 루프를
# 별도로 둡니다. 데이터 분할/스케일링/검증 풀/평가는 여전히 main.py
# 함수를 그대로 재사용해 ER/Naive와 같은 기준으로 비교되게 합니다.

def build_plain_loader(df_raw: pd.DataFrame, scaler, shuffle: bool) -> DataLoader:
    """df_raw(raw) → (X_model, y) 텐서로 변환한 뒤 plain DataLoader로 감쌉니다."""
    X_raw = df_raw[m.features].values.astype("float32")
    X_model = m.apply_scaler(X_raw, scaler)
    y = df_raw[m.target].values.astype("float32").reshape(-1, 1)
    ds = TensorDataset(torch.tensor(X_model, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    gen = m.dataloader_generator if shuffle else None
    return DataLoader(ds, batch_size=m.BATCH_SIZE, shuffle=shuffle,
                      pin_memory=False, generator=gen)


@torch.no_grad()
def compute_avg_loss_plain(model, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    for X, Y in loader:
        X, Y = X.to(m.device), Y.to(m.device)
        total += m.lfn.MSE_loss(model(X), Y).item()
    return total / max(len(loader), 1)


# ── EWC ────────────────────────────────────────────────────────────
LAMBDA_EWC = 600.0  # 정규화 강도. 너무 크면 새 태스크 학습이 거의 안 되고,
                     # 너무 작으면 정규화 효과가 사라져 Naive와 비슷해집니다.


def compute_fisher(model, loader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    이번 태스크 데이터에 대한 파라미터별 Fisher information을 근사
    계산합니다. 정확한 Fisher(로그우도의 2차 미분의 기댓값) 대신,
    "loss의 그래디언트 제곱의 평균"을 사용하는 empirical Fisher 근사를
    사용합니다(EWC 논문 및 다수 구현체에서 널리 쓰이는 근사법). 이
    값이 클수록 그 파라미t티를 줍니다.
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_seen = 0
    for X, Y in loader:
        X, Y = X.to(m.device), Y.to(m.device)
        model.zero_grad()
        loss = m.lfn.MSE_loss(model(X), Y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += (p.grad.detach() ** 2) * X.size(0)
        n_seen += X.size(0)
    for n in fisher:
        fisher[n] /= max(n_seen, 1)
    return fisher


def train_epoch_ewc(model, loader, optimizer,
                    fisher_dict: Optional[Dict[str, torch.Tensor]],
                    optpar_dict: Optional[Dict[str, torch.Tensor]]) -> float:
    model.train()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(m.device), Y.to(m.device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = m.lfn.MSE_loss(Y_pred, Y)

        # loss += lambda_ewc * sum(F_i * (theta_i - theta_i_old)^2)
        if fisher_dict is not None:
            penalty = torch.zeros((), device=m.device)
            for n, p in model.named_parameters():
                if n in fisher_dict:
                    penalty = penalty + (fisher_dict[n] * (p - optpar_dict[n]) ** 2).sum()
            loss = loss + LAMBDA_EWC * penalty

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── LwF ────────────────────────────────────────────────────────────
LAMBDA_LWF = 0.6  # distillation loss 가중치. main.py의 BETA(리플레이
                  # distillation 가중치)와 같은 역할을 하는 값입니다.


def train_epoch_lwf(model, teacher_model, loader, optimizer) -> float:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(m.device), Y.to(m.device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = m.lfn.MSE_loss(Y_pred, Y)

        if teacher_model is not None:
            with torch.no_grad():
                Y_teacher = teacher_model(X)
            # main.py의 distill_loss(= replay 샘플에 대해 teacher와
            # 맞추는 항)와 같은 컨셉을, "리플레이 샘플" 대신 "이번
            # 태스크의 현재 입력 X"에 대해 적용합니다. LwF의 핵심
            # 아이디어가 바로 이것입니다: 새 태스크 데이터를 넣어도
            # 예전 모델(teacher)의 출력을 크게 벗어나지 않게 만듭니다.
            distill = m.lfn.MSE_loss(Y_pred, Y_teacher)
            loss = loss + LAMBDA_LWF * distill

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ════════════════════════════════════════════════════════════════════
# EWC / LwF 공용: 한 태스크 학습
# ════════════════════════════════════════════════════════════════════
def train_one_task_custom(method: str, model, task_name: str,
                          train_ds_raw: pd.DataFrame,
                          fisher_dict=None, optpar_dict=None,
                          teacher_model=None):
    """
    main.py의 train_on_task()와 같은 역할을 하되, 리플레이 버퍼 없이
    EWC/LwF 방식으로 한 태스크를 학습합니다.

    - train/val 분할 : main.py와 동일하게 SEED, VAL_SIZE 사용
    - 스케일러       : main.py의 fit_task_scaler를 그대로 재사용합니다.
                       EWC/LwF는 memory_x(리플레이 버퍼)를 절대 채우지
                       않으므로("리플레이 버퍼 없이" 요구사항), 이 함수는
                       항상 "이번 태스크 데이터만으로 fit"과 동일하게
                       동작합니다.
    - 검증 누적 풀    : main.py의 accumulate_val_pool/build_val_pool_df를
                       그대로 재사용해 ER/Naive와 같은 기준으로 검증합니다.
    """
    inner_train, inner_val = m.train_test_split(
        train_ds_raw, test_size=m.VAL_SIZE, shuffle=True, random_state=m.SEED
    )
    inner_train = inner_train.reset_index(drop=True)
    inner_val = inner_val.reset_index(drop=True)

    scaler = m.fit_task_scaler(inner_train[m.features].values.astype("float32"))

    m.accumulate_val_pool(inner_val)
    val_pool_df = m.build_val_pool_df()

    train_loader = build_plain_loader(inner_train, scaler, shuffle=True)
    val_loader = build_plain_loader(val_pool_df, scaler, shuffle=False)

    optimizer = m.optim.Adam(model.parameters(), lr=m.LR)
    sched = m.CosineAnnealingLR(optimizer, T_max=m.NUM_EPOCHS, eta_min=1e-6)

    ckpt_path = os.path.join(m.save_model_dir, f"model_{method}_tmp.pth")
    best_val_loss = float("inf")

    for ep in range(1, m.NUM_EPOCHS + 1):
        if method == "EWC":
            train_loss = train_epoch_ewc(model, train_loader, optimizer,
                                         fisher_dict, optpar_dict)
        elif method == "LwF":
            train_loss = train_epoch_lwf(model, teacher_model, train_loader, optimizer)
        else:
            raise ValueError(f"알 수 없는 method: {method}")
        sched.step()

        val_loss = compute_avg_loss_plain(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            m.save_model(model, ckpt_path, epoch=ep, optimizer=optimizer, loss=best_val_loss)

        print(f"  [{method}] Epoch {ep:>3}/{m.NUM_EPOCHS} | Train Loss: {train_loss:.4f} "
             f"| Val Loss: {val_loss:.4f}")

    m.load_model(model, ckpt_path)
    return scaler


# ════════════════════════════════════════════════════════════════════
# 방법(method)별 continual 학습 실행
# ════════════════════════════════════════════════════════════════════
def run_experiment(method: str, raw_train: Dict[str, pd.DataFrame],
                   raw_test: Dict[str, pd.DataFrame]) -> List[Dict]:
    """
    지정한 method(ER/Naive/EWC/LwF)로 모든 태스크를 순서대로 학습하면서,
    main.py와 동일한 방식(매 stage마다 지금까지 학습한 모든 태스크의
    test set을 eval_task 단위로 개별 평가)으로 continual learning
    성능을 기록합니다. 평가는 항상 main.py의 evaluate()를 그대로
    재사용해서, 네 가지 방법의 MSE/RMSE/MAE 계산 방식이 완전히
    동일하도록 보장합니다.
    """
    reset_shared_state()
    m.set_seed(m.SEED)   # 방법마다 같은 seed에서 새로 시작 (공정 비교)

    model = fresh_model()
    fisher_dict, optpar_dict, teacher_model = None, None, None

    if method in ("ER", "Naive"):
        m.model = model
        m.USE_REPLAY = (method == "ER")

    records: List[Dict] = []
    seen_tasks: List[str] = []

    for stage, (file_name, train_data) in enumerate(raw_train.items()):
        if method in ("ER", "Naive"):
            # main.py를 거의 그대로 사용 (USE_REPLAY만 다름)
            m.train_on_task(train_data, task_name=file_name, val_size=m.VAL_SIZE)
            stage_scaler = m.CURRENT_SCALER
        else:
            stage_scaler = train_one_task_custom(
                method, model, file_name, train_data,
                fisher_dict=fisher_dict, optpar_dict=optpar_dict,
                teacher_model=teacher_model,
            )
            m.model = model  # evaluate()가 참조하는 전역 model을 맞춰줍니다.

            if method == "EWC":
                # 이번 태스크가 끝난 시점의 Fisher를 계산해 누적하고,
                # "old" 파라미터를 이번 태스크 학습 직후 값으로 갱신합니다.
                # (여러 태스크의 Fisher를 계속 더해가는 online-EWC 근사)
                fisher_loader = build_plain_loader(train_data, stage_scaler, shuffle=False)
                new_fisher = compute_fisher(model, fisher_loader)
                if fisher_dict is None:
                    fisher_dict = new_fisher
                else:
                    for n in fisher_dict:
                        fisher_dict[n] = fisher_dict[n] + new_fisher[n]
                optpar_dict = {n: p.clone().detach() for n, p in model.named_parameters()}
            elif method == "LwF":
                # 이번 태스크까지 학습한 모델을 고정된 teacher로 복제합니다.
                teacher_model = copy.deepcopy(model)
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad_(False)

        seen_tasks.append(file_name)
        for eval_task in seen_tasks:
            result = m.evaluate(f"[{method}] stage{stage}_{file_name}_on_{eval_task}",
                                raw_test[eval_task], scaler=stage_scaler)
            records.append({
                "method": method, "stage": stage, "stage_task": file_name,
                "eval_task": eval_task,
                "mse": result["mse"], "rmse": result["rmse"], "mae": result["mae"],
            })

    return records


# ════════════════════════════════════════════════════════════════════
# Average Accuracy / Backward Transfer(BWT) 계산
# ════════════════════════════════════════════════════════════════════
def compute_metrics(df: pd.DataFrame, method: str) -> Dict:
    """
    Average Accuracy와 Backward Transfer(BWT)를 계산합니다.
    (Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual
    Learning", NeurIPS 2017의 정의를 MSE 기준으로 적용한 버전입니다.)

    - Average Accuracy: 전체 태스크 학습이 끝난 시점(마지막 stage)에서,
      그때까지 학습한 모든 태스크의 test set에 대한 평균 MSE.
      (낮을수록 좋음)

    - BWT: 각 태스크 j(마지막으로 학습한 태스크는 제외)에 대해
          "그 태스크를 막 학습한 직후의 성능"(stage=j에서 eval_task=j)과
          "전체 학습이 끝난 뒤의 성능"(마지막 stage에서 eval_task=j)의
          차이를 구해 평균낸 값입니다.
              BWT_mse = mean_j [ MSE_final(j) - MSE_just_learned(j) ]
          (마지막으로 학습한 태스크는 그 이후 재학습이 없어 forgetting을
          측정할 수 없으므로 정의상 제외합니다.)

      ★ 주의: MSE는 낮을수록 좋은 지표이므로, 이 정의에서는 BWT_mse가
      양수일수록(0보다 클수록) forgetting이 심하다는 뜻입니다(다른
      태스크를 학습하는 동안 이 태스크의 오차가 커졌다는 뜻). 0에
      가까울수록, 또는 음수에 가까울수록(=오히려 나중 학습이 이전
      태스크 성능도 개선) forgetting이 적은 것입니다.

      ★★ 이 실험의 목적: ER의 |BWT_mse|가 Naive의 |BWT_mse|보다 훨씬
      작다는 것, 즉 ER의 BWT_mse가 0에 훨씬 더 가깝다는 것을 보여
      "리플레이 버퍼가 catastrophic forgetting을 완화한다"는 본 논문의
      핵심 주장을 뒷받침하는 것입니다.
    """
    sub = df[df["method"] == method].copy()
    task_order = (sub.drop_duplicates("stage_task")
                     .sort_values("stage")["stage_task"].tolist())
    last_stage = sub["stage"].max()
    final_rows = sub[sub["stage"] == last_stage]

    avg_accuracy = float(final_rows["mse"].mean())

    bwt_terms = []
    for stage_j, task_name in enumerate(task_order[:-1]):
        just_learned = sub[(sub["stage"] == stage_j) & (sub["eval_task"] == task_name)]
        final_perf = final_rows[final_rows["eval_task"] == task_name]
        if len(just_learned) and len(final_perf):
            bwt_terms.append(final_perf["mse"].values[0] - just_learned["mse"].values[0])

    bwt = float(np.mean(bwt_terms)) if bwt_terms else float("nan")

    return {
        "method": method,
        "num_tasks": len(task_order),
        "avg_accuracy_mse": avg_accuracy,
        "bwt_mse": bwt,
    }


# ════════════════════════════════════════════════════════════════════
# 시각화
# ════════════════════════════════════════════════════════════════════
def plot_cl_comparison(df: pd.DataFrame, freq_band: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(m.save_model_dir, "evaluation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    last_stage = df["stage"].max()
    final_df = df[df["stage"] == last_stage]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (좌) 최종 시점 평균 MSE (Average Accuracy) 비교: 네 방법의 절대 성능
    avg_by_method = final_df.groupby("method")["mse"].mean().sort_values()
    axes[0].bar(avg_by_method.index.astype(str), avg_by_method.values)
    axes[0].set_title(f"Average Accuracy (final MSE) - {freq_band}")
    axes[0].set_ylabel("MSE")

    # (우) 태스크별, stage 진행에 따른 forgetting 곡선(첫 번째 태스크 기준):
    # 네 방법이 이후 태스크들을 학습하는 동안 "가장 먼저 배운 태스크"의
    # 성능을 얼마나 유지하는지 보여줍니다. ER 곡선이 가장 평평할 것으로
    # 기대되는 그래프입니다.
    first_task = df["stage_task"].iloc[0]
    for method in df["method"].unique():
        sub = df[(df["method"] == method) & (df["eval_task"] == first_task)].sort_values("stage")
        axes[1].plot(sub["stage"], sub["mse"], marker="o", label=method)
    axes[1].set_title(f"Forgetting curve on first task ({first_task})")
    axes[1].set_xlabel("Stage (Trained Task Index)")
    axes[1].set_ylabel("MSE on first task's test set")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(plot_dir, f"cl_comparison_{freq_band}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[저장] {save_path}")


# ════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== CL 비교 실험 시작 (Naive vs ER vs EWC vs LwF) ===")
    print(f"주파수 대역: {m.FREQ_BAND}")

    # main.py와 완전히 동일한 CSV 생성/train-test 분할 (공정 비교의 전제)
    tasks = m.generate_csv_tasks(m.task_configs)
    raw_train: Dict[str, pd.DataFrame] = {}
    raw_test: Dict[str, pd.DataFrame] = {}
    for file_name, df in tasks:
        train_data, test_data = m.data_split(df)
        raw_train[file_name] = train_data
        raw_test[file_name] = test_data

    all_records: List[Dict] = []
    for method in ["Naive", "ER", "EWC", "LwF"]:
        print(f"\n########## [{method}] 학습 시작 ##########")
        all_records.extend(run_experiment(method, raw_train, raw_test))

    df_all = pd.DataFrame(all_records)
    df_all["freq_band"] = m.FREQ_BAND

    out_csv = os.path.join(m.save_model_dir, f"cl_comparison_{m.FREQ_BAND}.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\n[저장] {out_csv}")

    summary_rows = [compute_metrics(df_all, method) for method in df_all["method"].unique()]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(m.save_model_dir, f"cl_comparison_summary_{m.FREQ_BAND}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[저장] {summary_csv}")
    print("\n=== Average Accuracy / BWT 요약 (MSE 기준) ===")
    print(summary_df.to_string(index=False))

    plot_cl_comparison(df_all, m.FREQ_BAND)