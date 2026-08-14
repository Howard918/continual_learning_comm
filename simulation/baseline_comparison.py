"""
baseline_comparison.py
─────────────────────────────────────────────────────────────────────────
[우선순위 4] 회귀 모델 비교 (Joint training, non-continual)

※ 참고: 말씀하신 기존 baseline_comparison.py(LinearRegression, SVR,
MLP joint training 비교) 파일이 이번 업로드에는 포함되어 있지 않아,
요청하신 사양 — "같은 joint training 방식, 같은 scaler, 같은
train/test 데이터를 사용하고 RandomForestRegressor 행을 결과 CSV에
추가" — 에 맞춰 이 파일을 새로 작성했습니다. 기존 파일에 이 사양 외의
추가 로직(다른 feature 조합, 다른 평가 방식 등)이 있었다면 알려주시면
그대로 반영해서 다시 만들어 드리겠습니다.

Continual learning(순차 학습) 세팅과 별개로, "모든 태스크 데이터를
한 번에 모아(pooled) 학습했을 때" 여러 회귀 모델이 Path Loss를 얼마나
잘 예측하는지 비교하는 baseline입니다. LinearRegression / SVR /
RandomForestRegressor / MLP(main.py의 MyMLP, joint training) 네 가지를
같은 train/test 분할, 같은 scaler로 비교합니다.

이 실험은 forgetting 문제와는 무관하게, "애초에 이 문제(PL 예측)에
어떤 회귀 모델 계열이 적합한가"를 보여주기 위한 것입니다. 여기서 MLP를
joint(비-continual)로 학습한 결과는, cl_comparison.py의 ER/Naive
결과 대비 "continual 세팅이 joint 학습 대비 얼마나 손해를 보는지"를
가늠하는 참고선(upper-bound reference)으로도 사용할 수 있습니다.

결과:
    simulation/ER_/model/baseline_regression_comparison_{FREQ_BAND}.csv
        model, task, mse, rmse, mae
        (task="ALL"은 모든 태스크를 합친 pooled test 성능, 그 외에는
        태스크별 개별 test 성능입니다.)

실행 방법:
    python baseline_comparison.py
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

import main as m


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """main.py의 evaluate()와 동일한 정의(MSE 평균 → RMSE, 절대오차 평균 → MAE)."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
           "mae": float(np.mean(np.abs(y_true - y_pred)))}


def _pool_tasks(tasks):
    """
    모든 태스크의 train/test를 main.py와 동일한 방식(data_split, SEED,
    TEST_SIZE)으로 분할한 뒤 하나로 합칩니다. joint training이므로
    태스크 순서 구분 없이 한 번에 학습하지만, 태스크별 test 성능도
    함께 낼 수 있도록 __task__ 라벨 컬럼은 유지합니다.
    """
    train_frames, test_frames = [], []
    for file_name, df in tasks:
        train_df, test_df = m.data_split(df)     # main.py와 완전히 동일한 분할
        train_df = train_df.copy()
        train_df["__task__"] = file_name
        test_df = test_df.copy()
        test_df["__task__"] = file_name
        train_frames.append(train_df)
        test_frames.append(test_df)
    return pd.concat(train_frames, ignore_index=True), pd.concat(test_frames, ignore_index=True)


def _joint_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    main.py의 MyMLP를 continual 방식이 아니라, 모든 태스크 데이터를 한
    번에 섞어 학습(joint training)합니다. 태스크 순서 자체가 없으므로
    리플레이 버퍼/EWC/LwF 같은 continual learning 로직은 전혀 필요
    없고, 아주 단순한 표준 학습 루프만 사용합니다.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.utils import clip_grad_norm_

    m.set_seed(m.SEED)
    model = m.MyMLP(input_dim=4, hidden_dim=m.HIDDEN_DIM, output_dim=1,
                    layer_num=m.LAYER_NUM).to(m.device)
    optimizer = m.optim.Adam(model.parameters(), lr=m.LR)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=m.BATCH_SIZE,
                        shuffle=True, generator=m.dataloader_generator)

    model.train()
    for _ in range(m.NUM_EPOCHS):
        for X, Y in loader:
            X, Y = X.to(m.device), Y.to(m.device)
            optimizer.zero_grad()
            loss = m.lfn.MSE_loss(model(X), Y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_te = torch.tensor(X_test, dtype=torch.float32).to(m.device)
        pred = model(X_te).cpu().numpy().reshape(-1)
    return pred


def run() -> pd.DataFrame:
    print(f"=== 회귀 모델 baseline 비교 (joint training, {m.FREQ_BAND}) ===")
    tasks = m.generate_csv_tasks(m.task_configs)
    train_df, test_df = _pool_tasks(tasks)

    # main.py와 동일하게 USE_SCALING 플래그를 존중합니다.
    scaler = RobustScaler() if m.USE_SCALING else None
    X_train_raw = train_df[m.features].values.astype("float32")
    X_test_raw = test_df[m.features].values.astype("float32")
    if scaler is not None:
        scaler.fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    y_train = train_df[m.target].values.astype("float64")
    y_test = test_df[m.target].values.astype("float64")

    rows: List[Dict] = []

    def _eval_and_record(model_name: str, pred_all: np.ndarray) -> None:
        overall = _metrics(y_test, pred_all)
        overall.update({"model": model_name, "task": "ALL"})
        rows.append(overall)
        for task_name in test_df["__task__"].unique():
            mask = (test_df["__task__"] == task_name).values
            per_task = _metrics(y_test[mask], pred_all[mask])
            per_task.update({"model": model_name, "task": task_name})
            rows.append(per_task)

    # ── LinearRegression ──────────────────────────────────────────
    m.set_seed(m.SEED)
    lr = LinearRegression().fit(X_train, y_train)
    _eval_and_record("LinearRegression", lr.predict(X_test))

    # ── SVR ────────────────────────────────────────────────────────
    m.set_seed(m.SEED)
    svr = SVR(kernel="rbf").fit(X_train, y_train)
    _eval_and_record("SVR", svr.predict(X_test))

    # ── RandomForestRegressor (신규 추가) ────────────────────────────
    m.set_seed(m.SEED)
    rf = RandomForestRegressor(n_estimators=200, random_state=m.SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    _eval_and_record("RandomForest", rf.predict(X_test))

    # ── MLP (main.py의 MyMLP, joint training) ───────────────────────
    mlp_pred = _joint_mlp(X_train, y_train, X_test)
    _eval_and_record("MLP_joint", mlp_pred)

    out_df = pd.DataFrame(rows)[["model", "task", "mse", "rmse", "mae"]]
    out_path = os.path.join(m.save_model_dir, f"baseline_regression_comparison_{m.FREQ_BAND}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[저장] {out_path}")
    print("\n=== 전체(pooled) test 성능 비교 ===")
    print(out_df[out_df["task"] == "ALL"].sort_values("rmse").to_string(index=False))
    return out_df


if __name__ == "__main__":
    run()