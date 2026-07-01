"""
verify_pipeline.py
-------------------
지금까지 수정한 4가지 사항을 순서대로 검증하는 통합 테스트 스크립트.

검증 항목:
  [1] dted.py    : get_local_dted가 1차원(grid_lon/grid_lat)을 반환하는지
  [2] utility.py : get_v_factor(h=0)이 스칼라 -1을 반환하는지
  [3] 전체 격자  : get_received_power가 NaN 없이 정상 계산되는지 (49개 근거리 제외)
  [4] continual_trainer.py : ReservoirBuffer 균등성 + 옵티마이저 재바인딩(연속 fit) 검증

실행 위치: simulation/ 폴더 안에서 실행
    cd simulation
    python verify_pipeline.py

각 항목은 독립적으로 PASS/FAIL을 출력하고, 마지막에 전체 요약을 보여준다.
"""

import sys
import traceback
import numpy as np

RESULTS = {}  # {check_name: (passed: bool, message: str)}


def report(name, passed, message=""):
    RESULTS[name] = (passed, message)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" — {message}" if message else ""))


def section(title):
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════
# [1] dted.py — get_local_dted 1차원 반환 검증
# ═══════════════════════════════════════════════════════════════════
def check_1_dted_shape():
    section("[1] dted.py: get_local_dted가 1차원 배열을 반환하는지 검증")
    try:
        from map_factory.dted import get_local_dted

        dted_data = get_local_dted(127.0411, 36.0317, 1.0, 1.0)
        lon_shape = dted_data["grid_lon"].shape
        lat_shape = dted_data["grid_lat"].shape
        lon0 = dted_data["grid_lon"][0]

        print(f"  grid_lon.shape = {lon_shape}")
        print(f"  grid_lat.shape = {lat_shape}")
        print(f"  grid_lon[0] type = {type(lon0).__name__}, value = {lon0}")

        is_1d = (len(lon_shape) == 1) and (len(lat_shape) == 1)
        is_scalar = np.isscalar(lon0) or (hasattr(lon0, "shape") and lon0.shape == ())

        if is_1d and is_scalar:
            report("1. dted 1차원 반환", True,
                   f"grid_lon/lat이 1차원이고 인덱싱 시 스칼라 반환")
        else:
            report("1. dted 1차원 반환", False,
                   f"여전히 다차원이거나 인덱싱 결과가 배열입니다 (.ravel() 누락 의심)")
    except Exception as e:
        report("1. dted 1차원 반환", False, f"예외 발생: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# [2] utility.py — get_v_factor(h=0) 스칼라 검증
# ═══════════════════════════════════════════════════════════════════
def check_2_v_factor_scalar():
    section("[2] utility.py: get_v_factor(h=0)이 스칼라 -1을 반환하는지 검증")
    try:
        from utility.utility import get_v_factor

        v = get_v_factor(0, 1e8, 1000, 1000)
        v_np_scalar = get_v_factor(np.float64(0.0), 1e8, 1000.0, 1000.0)

        print(f"  get_v_factor(0, ...) = {v}  (type: {type(v).__name__})")
        print(f"  get_v_factor(np.float64(0.0), ...) = {v_np_scalar}  "
              f"(type: {type(v_np_scalar).__name__})")

        # 배열이 아니라 정수/실수 스칼라여야 함
        ok1 = not hasattr(v, "shape") or v.shape == ()
        ok2 = not hasattr(v_np_scalar, "shape") or v_np_scalar.shape == ()

        if ok1 and ok2 and v == -1 and v_np_scalar == -1:
            report("2. v_factor 스칼라 반환", True, "h=0일 때 스칼라 -1 반환 확인")
        else:
            report("2. v_factor 스칼라 반환", False,
                   "h=0 케이스가 배열을 반환합니다 (np.array([-1]) 의심)")
    except Exception as e:
        report("2. v_factor 스칼라 반환", False, f"예외 발생: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# [3] 전체 격자 — get_received_power NaN 비율 검증
# ═══════════════════════════════════════════════════════════════════
def check_3_full_grid_rp():
    section("[3] 전체 격자: get_received_power가 정상적으로 계산되는지 검증 (송신기 1개)")
    try:
        from map_factory.dted import get_local_dted
        from utility.utility import get_index
        from utility.calculator import get_received_power, get_info_about_observer_predicted_power

        t_lon, t_lat = 127.0411, 36.0317  # MiReuk
        span_lon, span_lat = 1.0, 1.0
        t_h, r_h = 10, 10
        f = 100_000_000

        dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
        t_ix, t_iy = get_index(dted_data, t_lon, t_lat)

        n_lat = len(dted_data["grid_lat"])
        n_lon = len(dted_data["grid_lon"])
        total = n_lat * n_lon

        rp_valid, rp_nan, errors = 0, 0, 0
        h_valid, h_nan = 0, 0

        for r_iy in range(n_lat):
            for r_ix in range(n_lon):
                try:
                    rp = get_received_power(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
                    info = get_info_about_observer_predicted_power(
                        dted_data, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
                    if np.isnan(rp):
                        rp_nan += 1
                    else:
                        rp_valid += 1
                    if np.isnan(info["H"]):
                        h_nan += 1
                    else:
                        h_valid += 1
                except Exception:
                    errors += 1

        valid_ratio = rp_valid / total
        print(f"  전체 격자: {total}개")
        print(f"  RP: valid={rp_valid} ({valid_ratio*100:.1f}%), nan={rp_nan}, errors={errors}")
        print(f"  H : valid={h_valid}, nan={h_nan}")

        # 근거리 필터(7x7=49칸 내외)를 제외하면 거의 전부 유효해야 함
        # 95% 이상 유효하고 런타임 에러가 0건이면 통과
        if errors == 0 and valid_ratio > 0.95:
            report("3. 전체 격자 RP 정상 계산", True,
                   f"유효 비율 {valid_ratio*100:.1f}%, 런타임 에러 0건")
        else:
            report("3. 전체 격자 RP 정상 계산", False,
                   f"유효 비율 {valid_ratio*100:.1f}% 또는 에러 {errors}건 발생")
    except Exception as e:
        report("3. 전체 격자 RP 정상 계산", False, f"예외 발생: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# [4] continual_trainer.py — Reservoir 균등성 + 옵티마이저 재바인딩 검증
# ═══════════════════════════════════════════════════════════════════
def check_4_reservoir_uniformity():
    section("[4-a] ReservoirBuffer: 균등 샘플링 분포 검증 (통계적 검정)")
    try:
        from continual_trainer import ReservoirBuffer
        import pandas as pd

        # 0~999번 ID를 가진 가짜 데이터 1000개를 max_size=100 버퍼에 적재
        N_TOTAL = 1000
        MAX_SIZE = 100
        buf = ReservoirBuffer(max_size=MAX_SIZE)

        df = pd.DataFrame({
            "R": list(range(N_TOTAL)),  # R값을 ID로 사용
            "D": [0.0] * N_TOTAL,
            "H": [0.0] * N_TOTAL,
            "F": [0.0] * N_TOTAL,
            "RP": [0.0] * N_TOTAL,
        })
        buf.add(df)

        print(f"  {N_TOTAL}개 중 버퍼 크기 {MAX_SIZE}로 Reservoir Sampling")
        print(f"  버퍼 상태: {buf}")

        survived_ids = sorted(int(r[0]) for r in buf.buffer)
        print(f"  생존한 ID 개수: {len(survived_ids)}")

        # 균등성 간이 검정: 생존 ID들이 0~999 전 구간에 고르게 퍼져있는지
        # (전반부/후반부에 쏠리지 않아야 함 — 단순 FIFO/큐였다면 후반부에만 쏠림)
        first_half = sum(1 for i in survived_ids if i < N_TOTAL // 2)
        second_half = len(survived_ids) - first_half
        ratio = min(first_half, second_half) / max(first_half, second_half) if max(first_half, second_half) > 0 else 0

        print(f"  전반부(ID<500) 생존: {first_half}개, 후반부(ID>=500) 생존: {second_half}개")
        print(f"  균형 비율 (작은쪽/큰쪽): {ratio:.2f}  (1.0에 가까울수록 균등)")

        # 길이가 정확히 MAX_SIZE인지 + 양쪽 절반에서 모두 일정 비율 이상 생존했는지
        ok_size = len(buf) == MAX_SIZE
        ok_balance = ratio > 0.5  # 한쪽이 다른쪽의 절반 이상이면 합격 (완전 쏠림 방지)

        if ok_size and ok_balance:
            report("4a. Reservoir 균등 샘플링", True,
                   f"크기 정확({MAX_SIZE}), 균형비율 {ratio:.2f}")
        else:
            report("4a. Reservoir 균등 샘플링", False,
                   f"크기={len(buf)} 또는 균형비율 {ratio:.2f}이 비정상적으로 한쪽에 쏠림")
    except Exception as e:
        report("4a. Reservoir 균등 샘플링", False, f"예외 발생: {e}")
        traceback.print_exc()

    section("[4-b] ContinualTrainer: 옵티마이저 재바인딩 후 연속 fit() 검증")
    try:
        from continual_trainer import build_trainer
        import pandas as pd

        trainer, buffer = build_trainer(buffer_size=5000)

        # 작은 가짜 새 데이터로 update()를 2번 연속 호출 — 두 번째 호출에서
        # "Unknown variable" 에러가 재현되는지가 핵심 검증 포인트
        fake_new_data = pd.DataFrame({
            "R": np.random.uniform(1000, 50000, 50),
            "D": np.random.uniform(100, 5000, 50),
            "H": np.random.uniform(-50, 50, 50),
            "F": [8.0] * 50,
            "RP": np.random.uniform(-140, -90, 50),
        })

        print("  1차 update() 호출...")
        h1 = trainer.update(fake_new_data, replay_ratio=0.5, epochs=1, batch_size=16)
        print("  1차 update() 성공")

        print("  2차 update() 호출 (여기서 기존에 'Unknown variable' 에러가 발생했음)...")
        h2 = trainer.update(fake_new_data, replay_ratio=0.5, epochs=1, batch_size=16)
        print("  2차 update() 성공")

        print("  3차 update() 호출 (연속 호출 안정성 재확인)...")
        h3 = trainer.update(fake_new_data, replay_ratio=0.5, epochs=1, batch_size=16)
        print("  3차 update() 성공")

        if h1 is not None and h2 is not None and h3 is not None:
            report("4b. 옵티마이저 재바인딩 (연속 fit)", True,
                   "update() 3회 연속 호출 모두 성공, Unknown variable 에러 없음")
        else:
            report("4b. 옵티마이저 재바인딩 (연속 fit)", False,
                   "update()가 None을 반환함 (가드에 걸렸을 가능성)")
    except ValueError as e:
        if "Unknown variable" in str(e):
            report("4b. 옵티마이저 재바인딩 (연속 fit)", False,
                   f"원래 버그가 재현되었습니다: {e}")
        else:
            report("4b. 옵티마이저 재바인딩 (연속 fit)", False, f"다른 ValueError: {e}")
        traceback.print_exc()
    except Exception as e:
        report("4b. 옵티마이저 재바인딩 (연속 fit)", False, f"예외 발생: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Continual Learning 파이프라인 통합 검증을 시작합니다.")
    print("(simulation/ 폴더 안에서 실행되어야 상대경로가 맞습니다)")

    check_1_dted_shape()
    check_2_v_factor_scalar()
    check_3_full_grid_rp()
    check_4_reservoir_uniformity()

    section("최종 요약")
    all_passed = True
    for name, (passed, msg) in RESULTS.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(">>> 모든 검증 항목을 통과했습니다.")
        sys.exit(0)
    else:
        print(">>> 일부 항목이 실패했습니다. 위 로그를 확인하세요.")
        sys.exit(1)