"""
compare_models.py
-------------------
기존 모델(main.h5)과 continual learning으로 업데이트된 모델(continual_main.h5)의
성능을 동일 기준으로 비교하는 스크립트.

평가 기준: REAL(물리 모델, get_received_power)을 정답으로 보고
           각 MLP 모델의 예측값과의 절댓값 오차 평균(MAE)을 비교한다.

비교는 두 그룹의 송신기에 대해 따로 수행한다.
  - 기존 송신기 (all_transmitters 중 continual 학습에 쓰지 않은 것들)
    → 두 모델 모두 학습에 쓰인 데이터. "기존 지식을 유지했는가"를 본다.
  - 새 송신기 (continual_trainer.py에서 새로 학습시킨 것들)
    → main.h5는 본 적 없는 데이터. "새 지식을 잘 배웠는가"를 본다.

이 두 그룹을 나눠서 보는 이유: continual learning의 핵심 평가축은
"새 데이터에 적응했는가"와 "기존 데이터를 잊지 않았는가"(Catastrophic
Forgetting 여부) 두 가지이기 때문이다. 전체 평균만 보면 이 둘이 섞여서
드러나지 않는다.

실행 위치: simulation/ 폴더 안에서 실행
    cd simulation
    python compare_models.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
from math import log10
from tensorflow import keras

from map_factory.dted import get_local_dted
from utility.utility import get_index, convert_to_si
from utility.calculator import get_received_power, get_info_about_observer_predicted_power
from environments.transmitter import all_transmitters, test_transmitters, transmitters

# ── 비교 대상 모델 경로 ────────────────────────────────────────────────
MODEL_PATHS = {
    "main (기존)":      ("MLP/MODELS/main.h5",           "MLP/MLP.scaler.gz"),
    "continual_main (신규)": ("MLP/MODELS/continual_main.h5", "MLP/MLP.continual.scaler.gz"),
}

# 평가에 쓸 송신기와 주파수
FREQUENCY = 100_000_000
R_H = 10


def evaluate_model_on_transmitter(model, scaler, transmitter, frequency=FREQUENCY):
    """
    하나의 (모델, 송신기) 조합에 대해 REAL 대비 MAE를 계산.

    Returns
    -------
    dict: {"MAE": float, "n_valid": int, "n_total": int}
    """
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)

    n_lat = len(dted_data["grid_lat"])
    n_lon = len(dted_data["grid_lon"])

    errors = []
    for r_iy in range(n_lat):
        for r_ix in range(n_lon):
            # REAL: 물리 모델 (정답)
            real_rp = get_received_power(
                dted_data, frequency, t_ix, t_iy, t_h, r_ix, r_iy, R_H)
            if np.isnan(real_rp):
                continue

            # PRED: MLP 모델 예측
            info = get_info_about_observer_predicted_power(
                dted_data, t_ix, t_iy, t_h, r_ix, r_iy, R_H)
            if np.isnan(info["H"]):
                continue

            try:
                X = pd.DataFrame(
                    [[info["R"], info["D"], info["H"], log10(frequency)]],
                    columns=["R", "D", "H", "F"]
                )
                X_scaled = scaler.transform(X)
                pred_rp = model.predict(X_scaled, verbose=0)[0][0]
            except Exception:
                continue

            errors.append(abs(real_rp - pred_rp))

    n_valid = len(errors)
    n_total = n_lat * n_lon
    mae = float(np.mean(errors)) if n_valid > 0 else float("nan")

    return {"MAE": mae, "n_valid": n_valid, "n_total": n_total}


def run_comparison(transmitter_groups, frequency=FREQUENCY):
    """
    transmitter_groups: {"그룹이름": [transmitter, ...]} 형태.
    각 모델 × 각 그룹 × 각 송신기에 대해 MAE를 계산해 DataFrame으로 반환.
    """
    rows = []

    for model_label, (model_path, scaler_path) in MODEL_PATHS.items():
        print(f"\n{'='*70}")
        print(f"모델 로드: {model_label}  ({model_path})")
        print(f"{'='*70}")
        try:
            model = keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
        except (OSError, IOError) as e:
            print(f"  모델/스케일러 로드 실패: {e}")
            print(f"  {model_label}을(를) 건너뜁니다.")
            continue

        for group_name, transmitters_in_group in transmitter_groups.items():
            for transmitter in transmitters_in_group:
                t_name = transmitter.name
                print(f"  [{group_name}] {t_name} 평가 중 ({convert_to_si(frequency)}Hz)...")
                result = evaluate_model_on_transmitter(model, scaler, transmitter, frequency)
                print(f"    MAE = {result['MAE']:.3f} dB  "
                      f"(유효 격자 {result['n_valid']}/{result['n_total']})")
                rows.append({
                    "model": model_label,
                    "group": group_name,
                    "transmitter": t_name,
                    "MAE_dB": result["MAE"],
                    "n_valid": result["n_valid"],
                    "n_total": result["n_total"],
                })

    return pd.DataFrame(rows)


def summarize(df):
    """모델 × 그룹별 평균 MAE를 요약 출력."""
    print(f"\n{'='*70}")
    print("요약: 모델 × 그룹별 평균 MAE (dB, 낮을수록 좋음)")
    print(f"{'='*70}")

    summary = df.groupby(["group", "model"])["MAE_dB"].mean().unstack("model")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*70}")
    print("해석 가이드")
    print(f"{'='*70}")
    print("- '기존 송신기' 그룹에서 continual_main의 MAE가 main과 비슷하거나 낮다")
    print("  → Catastrophic Forgetting 없이 기존 지식을 유지함 (Experience Replay 효과)")
    print("- '기존 송신기' 그룹에서 continual_main의 MAE가 크게 증가했다")
    print("  → 기존 지식을 잊음. replay_ratio를 높이거나 buffer_size를 키워야 함")
    print("- '새 송신기' 그룹에서 continual_main의 MAE가 main보다 크게 낮다")
    print("  → 새 지역에 성공적으로 적응함 (Continual Learning 효과)")
    print("- '새 송신기' 그룹에서 두 모델의 MAE가 비슷하다")
    print("  → 새 지역 학습이 거의 반영되지 않음. epochs를 늘리거나 학습률을 확인")

    return summary


if __name__ == "__main__":
    # 평가 그룹 구성:
    #   "기존 송신기" = 원래 main.h5 학습에 쓰인 4개 (transmitters)
    #   "새 송신기"   = continual_trainer.py에서 새로 학습시킨 송신기들
    #                  (실제 실행한 new_transmitters/test_transmitters로 바꿔서 사용)
    transmitter_groups = {
        "기존 송신기": transmitters,        # MiReuk, MuDeung, SikJang, NamSan
        "새 송신기":   test_transmitters,   # ChungJu, Daegu, GIST (continual로 새로 학습시킨 대상)
    }

    df = run_comparison(transmitter_groups)

    if df.empty:
        print("\n비교할 결과가 없습니다. 모델 경로를 확인하세요.")
        sys.exit(1)

    df.to_csv("model_comparison_result.csv", index=False)
    print(f"\n상세 결과 저장: model_comparison_result.csv")

    summarize(df)