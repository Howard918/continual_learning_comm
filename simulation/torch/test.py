import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 재현성을 위한 시드 고정
torch.manual_seed(42)

# ------------------------------
# 1. 더미 데이터 생성
# ------------------------------
n_samples = 1000
n_features = 4

X = torch.randn(n_samples, n_features)

# 임의의 가중치와 편향을 이용해 타겟(y) 생성 + 약간의 노이즈 추가
true_weights = torch.tensor([2.0, -1.0, 0.5, 3.0])
true_bias = 1.0
noise = 0.1 * torch.randn(n_samples)

y = X @ true_weights + true_bias + noise
y = y.unsqueeze(1)  # (n_samples, 1) 형태로 변환

# 학습/검증 데이터 분할
n_train = 800
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

# ------------------------------
# 2. MLP 모델 정의
# ------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    # forward() 오버라이딩
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=n_features, hidden_dim=64, output_dim=1)

# ------------------------------
# 3. 손실 함수 및 옵티마이저
# ------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ------------------------------
# 4. 학습 루프
# ------------------------------
n_epochs = 200

# Full-Batch Gradient Descent
for epoch in range(n_epochs):
    # 모델 학습모드
    model.train()

    # 이전 스텝에서 계산된 그래디언트 초기화
    optimizer.zero_grad()

    # 모델 예측값 계산
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 역전파 알고리즘을 통해 그래디언트 계산
    loss.backward()

    # 계산된 그래디언트를 이용해 모델 파라미터(w, b) 업데이트
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        # 모델 평가모드
        model.eval()

        # 검증 단계에서는 그래디언트 계산하지 않음(메모리 절약, 속도 향상)
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        print(f"Epoch [{epoch+1}/{n_epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# ------------------------------
# 5. 예측 테스트
# ------------------------------
# 모델 평가모드
model.eval()
with torch.no_grad():
    sample = X_val[:5]
    pred = model(sample)
    print("\n예측값:\n", pred.squeeze())
    print("실제값:\n", y_val[:5].squeeze())


# ------------------------------
# 6. 특징별 실제값 vs 예측값 시각화
# ------------------------------
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val)

# 텐서를 numpy로 변환 (플롯을 위해)
X_val_np = X_val.numpy()
y_val_np = y_val.numpy().flatten()
y_pred_np = y_pred_val.numpy().flatten()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(n_features):
    ax = axes[i]
    ax.scatter(X_val_np[:, i], y_val_np, alpha=0.5, label='actual', color='steelblue')
    ax.scatter(X_val_np[:, i], y_pred_np, alpha=0.5, label='predicted', color='orangered')
    ax.set_xlabel(f'Feature {i+1}')
    ax.set_ylabel('y')
    ax.set_title(f'Feature {i+1} vs y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------
# 7. 예측값 vs 실제값 (전체적인 정확도 확인)
# ------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_val_np, y_pred_np, alpha=0.5, color='seagreen')
min_val = min(y_val_np.min(), y_pred_np.min())
max_val = max(y_val_np.max(), y_pred_np.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = y_hat')
plt.xlabel('actual y')
plt.ylabel('predicted y_hat')
plt.title('predicted vs actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()