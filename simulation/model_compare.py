"""
첫 번째 학습 태스크(stage 0에서 학습된 태스크)에 대한 Test 성능을,
ER(Experience Replay) 사용 여부에 따라 비교하는 스크립트.

입력:
  - results_ER_MLP_..._LTE_2.1G.csv   (use_replay=True)
  - results_MLP_..._LTE_2.1G.csv      (use_replay=False)

두 파일 모두 main.py가 생성하는 results_*.csv 포맷을 따른다:
  run_name, sweep_tag, use_replay, alpha, hidden_dim, layer_num,
  freq_band, stage, stage_task, eval_task, mse, rmse, mae

동작:
  1) stage=0에서 학습된 태스크(stage_task)를 "첫 번째 태스크"로 자동 식별
  2) 두 CSV에서 eval_task == 첫 번째 태스크인 행만 추출
     (= 이후 태스크들을 계속 학습하는 동안 첫 태스크의 성능이 어떻게 변하는지)
  3) stage에 따른 MSE/RMSE를 ER 모델과 Non-ER 모델로 나란히 비교 플롯
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ---- 입력 파일 경로 (필요시 수정) ----
ER_CSV = "results_ER_2_sweep_alpha0.5_hd64_layer2.csv"
NOER_CSV = "results_MLP_2_sweep_alpha0.5_hd64_layer2.csv"
OUTPUT_PNG = "2_first_task_er_vs_noer.png"


def load_first_task_curve(csv_path: str) -> pd.DataFrame:
    """CSV를 읽어 '첫 번째 학습 태스크'에 대한 stage별 성능만 추출한다."""
    df = pd.read_csv(f"simulation/ER_/model/{csv_path}")
    first_task = df.loc[df["stage"] == 0, "stage_task"].iloc[0]
    curve = df[df["eval_task"] == first_task].sort_values("stage")
    return curve, first_task

def main():
    er_curve, first_task_er = load_first_task_curve(ER_CSV)
    noer_curve, first_task_noer = load_first_task_curve(NOER_CSV)

    assert first_task_er == first_task_noer, (
        f"두 파일의 첫 번째 태스크가 다릅니다: {first_task_er} vs {first_task_noer}"
    )
    first_task = first_task_er

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in zip(axes, ["mse", "rmse"], ["MSE", "RMSE"]):
        ax.plot(er_curve["stage"], er_curve[metric], marker="o",
                 label="ER (Experience Replay)", color="tab:orange")
        ax.plot(noer_curve["stage"], noer_curve[metric], marker="o",
                 label="No ER (Naive)", color="tab:blue")
        ax.set_xlabel("Training Stage")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Test {ylabel} on first task\n({first_task})")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"저장 완료 -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()