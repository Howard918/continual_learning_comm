"""
continual_trainer.py
--------------------
Reservoir Sampling 기반 Experience Replay + Continual Learning

흐름:
  1. data.csv → ReservoirBuffer 초기 적재 (기존 데이터)
  2. create_CSV()로 새 송신기 데이터 생성
  3. ContinualTrainer.update() 호출
     - 새 데이터를 Reservoir Sampling으로 버퍼에 편입
     - 버퍼 샘플 + 새 데이터 혼합 → 점진적 학습
     - scaler를 혼합 데이터로 재피팅 후 저장
  4. 학습된 모델 저장
"""

import random
import numpy as np
import pandas as pd
from math import log10
import joblib
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# ── 경로 상수 ──────────────────────────────────────────────────────────
EXISTING_CSV    = "MLP/DATA/data.csv"
MODEL_PATH      = "MLP/MODELS/main.h5"
SCALER_PATH     = "MLP/MLP.scaler.gz"
UPDATED_MODEL   = "MLP/MODELS/continual_main.h5"
UPDATED_SCALER  = "MLP/MLP.continual.scaler.gz"
COLS            = ["R", "D", "H", "F", "RP"]   # csv_factory가 생성하는 컬럼 순서


# ══════════════════════════════════════════════════════════════════════
# 1. Reservoir Sampling Buffer
# ══════════════════════════════════════════════════════════════════════

class ReservoirBuffer:
    """
    고정 크기 버퍼에 스트리밍 데이터를 균등 확률로 보존하는 Reservoir Sampling.

    알고리즘 (Algorithm R, Vitter 1985):
      - 버퍼가 아직 가득 차지 않았으면 그냥 추가
      - 가득 찬 이후 i번째 샘플은 확률 max_size/i 로 버퍼 내 임의 위치와 교체

    이 방식을 쓰는 이유:
      - 전체 데이터를 메모리에 올리지 않아도 됨
      - 어느 시점에 sampling해도 지금까지 본 모든 샘플에 대해 균등 확률 보장
      - 오래된 데이터가 자동으로 밀려나지 않고 균등하게 생존 → 과거 지식 보존
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size   = max_size
        self.buffer     = []          # list of (R, D, H, F, RP)
        self.n_seen     = 0           # 지금까지 add()에 넣은 총 샘플 수

    # ── 단일 행 추가 ──────────────────────────────────────────────────
    def _add_one(self, row: tuple):
        self.n_seen += 1
        if len(self.buffer) < self.max_size:
            # 버퍼가 가득 차지 않았으면 무조건 추가
            self.buffer.append(row)
        else:
            # Reservoir Sampling: i번째 샘플을 확률 max_size/n_seen 으로 교체
            j = random.randint(0, self.n_seen - 1)
            if j < self.max_size:
                self.buffer[j] = row

    # ── DataFrame 단위 추가 ───────────────────────────────────────────
    def add(self, df: pd.DataFrame):
        """DataFrame의 각 행을 순서대로 Reservoir Sampling으로 버퍼에 편입."""
        for row in df[COLS].itertuples(index=False):
            self._add_one(tuple(row))

    # ── 샘플링 ────────────────────────────────────────────────────────
    def sample(self, n: int) -> pd.DataFrame:
        """버퍼에서 n개를 비복원 랜덤 샘플링해 DataFrame으로 반환."""
        n = min(n, len(self.buffer))
        rows = random.sample(self.buffer, n)
        return pd.DataFrame(rows, columns=COLS)

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return (f"ReservoirBuffer(max_size={self.max_size}, "
                f"stored={len(self.buffer)}, seen={self.n_seen})")


# ══════════════════════════════════════════════════════════════════════
# 2. ContinualTrainer
# ══════════════════════════════════════════════════════════════════════

class ContinualTrainer:
    """
    Experience Replay + Continual Learning 트레이너.

    update() 한 번의 흐름:
      1. 새 데이터를 버퍼에 Reservoir Sampling으로 편입
      2. 버퍼에서 replay_ratio 비율만큼 과거 샘플 추출
      3. 새 데이터 + 과거 샘플 혼합 → 셔플
      4. 혼합 데이터로 scaler 재피팅 (범위 갱신)
      5. model.fit()으로 점진적 학습 (기존 가중치에서 이어서)
      6. 모델·scaler 저장
    """

    def __init__(
        self,
        model,
        scaler,
        buffer: ReservoirBuffer,
        model_save_path: str = UPDATED_MODEL,
        scaler_save_path: str = UPDATED_SCALER,
    ):
        self.model            = model
        self.scaler           = scaler
        self.buffer           = buffer
        self.model_save_path  = model_save_path
        self.scaler_save_path = scaler_save_path
        self.update_count     = 0     # 몇 번 update()가 호출됐는지 기록

    # ── 핵심 메서드 ───────────────────────────────────────────────────
    def update(
        self,
        new_data: pd.DataFrame,
        replay_ratio: float = 0.5,
        epochs: int = 50,
        batch_size: int = 32,
        val_split: float = 0.1,
    ):
        """
        Parameters
        ----------
        new_data     : create_CSV가 생성한 새 송신기의 DataFrame (COLS 컬럼 포함)
        replay_ratio : 새 데이터 대비 과거 샘플 비율 (0.5 → 새 데이터와 1:1 혼합)
        epochs       : 점진적 학습 epoch 수
        batch_size   : 미니배치 크기
        val_split    : 혼합 데이터 중 검증셋 비율
        """
        self.update_count += 1
        print(f"\n[Update #{self.update_count}] 새 데이터: {len(new_data)}행  "
              f"버퍼 현황: {self.buffer}")

        # Step 1. 새 데이터 → 버퍼 편입 (Reservoir Sampling)
        self.buffer.add(new_data)
        print(f"  버퍼 편입 후: {self.buffer}")

        # Step 2. 버퍼에서 과거 샘플 추출
        n_replay = int(len(new_data) * replay_ratio)
        replay_data = self.buffer.sample(n_replay)
        print(f"  과거 샘플: {len(replay_data)}행 (replay_ratio={replay_ratio})")

        # Step 3. 새 데이터 + 과거 샘플 혼합 → 셔플
        mixed = (
            pd.concat([new_data[COLS], replay_data], ignore_index=True)
            .sample(frac=1)
            .reset_index(drop=True)
        )
        print(f"  혼합 데이터: {len(mixed)}행")

        # Step 4. scaler 재피팅 (새 데이터 범위를 반영)
        X_mixed = mixed[["R", "D", "H", "F"]]
        y_mixed = mixed["RP"].values

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(X_mixed)
        X_scaled = self.scaler.transform(X_mixed)

        # Step 5. 점진적 학습 (기존 가중치 유지, 이어서 학습)
        #
        # load_model()로 불러온 모델의 옵티마이저는 로드 시점의 변수에 바인딩되어
        # 있어서, 이후 fit()을 반복 호출하면 (특히 update()가 여러 번 불릴 때)
        #   "Unknown variable ... This optimizer can only be called for the
        #    variables it was originally built with."
        # 에러가 발생할 수 있다. fit() 직전에 동일한 옵티마이저 설정으로
        # 명시적으로 재컴파일해 옵티마이저를 현재 모델 변수에 다시 묶어준다.
        self.model.compile(
            optimizer=(
                self.model.optimizer.__class__.from_config(self.model.optimizer.get_config())
                if self.model.optimizer is not None else "adam"
            ),
            loss=self.model.loss if self.model.loss is not None else "mse",
        )

        history = self.model.fit(
            X_scaled, y_mixed,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=1,
        )

        # Step 6. 저장
        self.model.save(self.model_save_path)
        joblib.dump(self.scaler, self.scaler_save_path)
        print(f"  모델 저장: {self.model_save_path}")
        print(f"  scaler 저장: {self.scaler_save_path}")

        # 마지막 epoch 손실 출력
        final_loss     = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        print(f"  최종 loss: {10*log10(final_loss):.3f} dB  "
              f"val_loss: {10*log10(final_val_loss):.3f} dB")

        return history


# ══════════════════════════════════════════════════════════════════════
# 3. 초기화 헬퍼
# ══════════════════════════════════════════════════════════════════════

def load_existing_data(csv_path: str = EXISTING_CSV) -> pd.DataFrame:
    """
    기존 data.csv를 로드하고 컬럼을 COLS 순서로 정리해 반환.
    additonal_training.py와 동일하게 'Unnamed: 0' 컬럼을 제거.
    """
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # F 컬럼이 없으면 (구버전 csv) 경고
    if "F" not in df.columns:
        raise ValueError("data.csv에 F 컬럼이 없습니다. csv_factory를 재실행하세요.")
    return df[COLS]


def build_trainer(
    buffer_size: int = 100_000,
    model_path: str  = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
) -> tuple[ContinualTrainer, ReservoirBuffer]:
    """
    기존 data.csv → 버퍼 초기 적재 후 ContinualTrainer 반환.

    Returns
    -------
    trainer : ContinualTrainer
    buffer  : ReservoirBuffer  (참조용; trainer.buffer와 동일 객체)
    """
    # 기존 데이터 로드
    existing = load_existing_data()
    print(f"기존 데이터 로드: {len(existing)}행")

    # 버퍼 초기화 및 기존 데이터 적재
    buffer = ReservoirBuffer(max_size=buffer_size)
    buffer.add(existing)
    print(f"버퍼 초기 적재 완료: {buffer}")

    # 모델·scaler 로드
    model  = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print(f"모델 로드: {model_path}")
    print(f"scaler 로드: {scaler_path}  feature_range={scaler.feature_range}")

    # load_model 직후 옵티마이저가 변수에 제대로 바인딩되지 않은 상태로
    # 올 수 있으므로(특히 .h5 + 최신 Keras 조합), 로드 시점에 한 번
    # 동일한 설정으로 재컴파일해 둔다.
    if model.optimizer is not None:
        model.compile(
            optimizer=model.optimizer.__class__.from_config(model.optimizer.get_config()),
            loss=model.loss if model.loss is not None else "mse",
        )
    else:
        # 컴파일 정보 없이 저장된 모델인 경우 기본값으로 컴파일
        print("  모델에 옵티마이저 정보가 없어 기본값(adam/mse)으로 컴파일합니다.")
        model.compile(optimizer="adam", loss="mse")

    trainer = ContinualTrainer(model, scaler, buffer)
    return trainer, buffer


# ══════════════════════════════════════════════════════════════════════
# 4. 실행 예시
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # csv_factory의 create_CSV와 transmitter 객체를 import
    # (실제 실행 시 simulation/ 폴더에서 실행해야 경로가 맞음)
    from csv_factory import create_CSV
    from environments.transmitter import new_transmitters  # 새로 추가할 송신기 목록

    NEW_FREQUENCY_LIST = [100_000_000]  # 새로 시뮬레이션할 주파수

    # ── Step 1. 기존 data.csv로 버퍼·트레이너 초기화 ──────────────────
    trainer, buffer = build_trainer(buffer_size=100_000)

    # ── Step 2. 새 송신기마다 데이터 생성 → 점진적 학습 ──────────────
    for transmitter in new_transmitters:
        name = transmitter.name
        print(f"\n{'='*60}")
        print(f"새 송신기: {name}")
        print(f"{'='*60}")

        # create_CSV를 단일 송신기에 대해 호출해 DataFrame 반환
        # (csv_factory의 create_CSV는 파일을 저장하지만 여기서는 반환값을 씀)
        new_df = create_CSV([transmitter], NEW_FREQUENCY_LIST, save_path=None)

        # ── Step 3. update(): Reservoir Sampling + 점진적 학습 ────────
        trainer.update(
            new_data     = new_df,
            replay_ratio = 0.5,   # 새 데이터 : 과거 샘플 = 1 : 0.5
            epochs       = 50,
            batch_size   = 32,
        )

    print("\n모든 송신기 업데이트 완료")
    print(f"최종 버퍼 현황: {buffer}")