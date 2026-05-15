# Recurrent VLA-JEPA

> **Latent Space Chain-of-Thought: 월드 모델 예측 오차를 QwenVL 추론 시퀀스에 통합하여 "반성하고 행동하는" 로봇 제어를 실현하는 VLA 프레임워크**

이 레포지토리는 [VLA-JEPA](https://arxiv.org/abs/2602.10098) (Sun et al., 2026)를 기반으로, V-JEPA2 월드 모델의 **예측 오차(Δz)**를 QwenVL의 토큰 생성 시퀀스에 직접 주입하는 새로운 추론 패러다임을 구현합니다.

---

## 1. 연구 동기: 학습과 추론의 불일치

VLA-JEPA는 학습 시 V-JEPA2 예측기(world model)를 사용해 미래 시각 토큰을 예측하고 이를 정책 학습에 활용한다. 그러나 **추론 시에는 예측기가 완전히 무시**되고, QwenVL이 현재 프레임만 처리한다.

```
[학습]  vj_encoder → vj_predictor → QwenVL → DiT
[추론]  vj_encoder              → QwenVL → DiT   ← predictor 누락
```

기존 VLA-JEPA가 `이미지 → 액션`의 직관적 매핑에 의존했다면, 본 연구는 **"예측(Hypothesis) → 오차 분석(Reflection) → 수정된 액션(Action)"** 의 추론 과정을 LLM의 잠재 공간(Latent Space) 내에서 시퀀스로 구현하는 것을 목표로 합니다.

---

## 2. 아키텍처: Latent Space Chain-of-Thought

### 핵심 아이디어

QwenVL의 자기 회귀(Autoregressive) 생성 순서를 재구성하여 월드 모델의 피드백을 직접 참조하게 합니다.

```
기존 QwenVL 시퀀스:
[image | instruction | action_tokens | embodied_action_tokens]
                           ↑ V-JEPA용          ↑ DiT용

새로운 QwenVL 시퀀스 (Stage 3):
[image | instruction | action_tokens | correction_tokens | embodied_action_tokens]
                           ↑ V-JEPA용       ↑ Δz 주입          ↑ DiT용
```

### 세 단계 추론 루프 ("Thinking Before Acting")

| Phase | 토큰 | 역할 |
|-------|------|------|
| **A — Hypothesis** | `action_tokens` | V-JEPA Predictor 조건부 입력. "이 액션을 하면 미래가 어떻게 될까?" |
| **B — Reflection** | `correction_tokens` | CorrectionProjector(Δz) 주입. "예측이 얼마나 틀렸는가?" |
| **C — Final Intent** | `embodied_action_tokens` | A와 B를 attention으로 참조해 생성. DiT 액션 헤드의 입력. |

`embodied_action_tokens`는 QwenVL의 attention을 통해 correction_tokens를 자연스럽게 참조하므로, 별도의 fusion 모듈 없이 예측 오차가 최종 액션 생성에 반영됩니다.

### 데이터 흐름

```
현재 관측  → V-JEPA Encoder  → z_obs_t  (현재 시각 특징)
이전 상태  → V-JEPA Predictor → ẑ_t     (이전 스텝의 예측)

Δz = z_obs_t - ẑ_t              (예측 오차)

CorrectionProjector(Δz)
  vj_dim(2816) → LayerNorm → Linear → qwen_dim(2048)
  256 spatial tokens → group mean-pool → 8 correction tokens

QwenVL([image | instruction | action_tokens | correction_tokens | emb_action_tokens])
  → embodied_action_tokens (Δz를 참조하여 생성)

DiT (frozen) → 7-DoF 액션 청크
```

---

## 3. 핵심 수식

### 예측 오차 계산

$$\Delta z_t = z_{obs,t} - \hat{z}_t$$

$$\hat{z}_t = \text{Predictor}(z_{obs,t-1},\ a_{t-1})$$

V-JEPA Predictor는 **이전 관측과 이전 액션**만을 입력으로 받습니다. Δz는 Predictor 자체를 수정하지 않고, 오직 QwenVL의 correction token 주입에만 사용됩니다.

### CorrectionProjector

$$\text{corr\_embed} = \sigma(\text{gate}) \cdot W \cdot \text{LayerNorm}(\text{GroupPool}(\Delta z))$$

- `gate` 초기값 = −4 → sigmoid(−4) ≈ 0.018: 초기에는 correction 신호가 거의 0
- cold start (t=0): Δz = 0 → correction_embed ≈ 0 → 기존 baseline과 동일하게 동작

---

## 4. 단계별 학습 로드맵

| Stage | 학습 대상 | 목적 |
|-------|-----------|------|
| **Stage 1** (저자 제공) | V-JEPA2 Encoder + Predictor | 세계 모델 사전 학습 |
| **Stage 2** (저자 제공) | QwenVL + DiT | 로봇 행동 데이터 파인튜닝 |
| **Stage 3** (본 연구) | **CorrectionProjector** | Latent CoT 주입; 예측 오차를 액션 생성에 반영 |

Stage 3에서 QwenVL과 DiT는 **완전히 frozen** 상태로 유지됩니다. CorrectionProjector(~10M 파라미터)만 학습합니다.

### Sanity Check (Stage 1 후)

V-JEPA Predictor의 유효성을 cos_sim 로깅으로 확인합니다:

```bash
bash examples/LIBERO/eval_libero_recurrent.sh libero_spatial 50
# 서버 로그에서 pred/obs cosine_sim 모니터링
```

---

## 5. 평가 전략: ABC → DE 공간적 외삽

로봇이 특정 위치를 암기하는 현상을 타파하기 위해, LIBERO-PRO 환경에서 학습/평가 위치를 분리합니다.

- **Training (ABC)**: 물체를 작업 영역의 일부 위치(A, B, C)에만 배치하여 학습
- **Testing (DE)**: 학습 시 전혀 보지 못한 새로운 위치(D, E)에서 평가

Δz 기반 오차 보정이 위치 외삽 시에도 작동하는지 검증하는 것이 핵심 실험입니다.

---

## 6. 독창적 가치 (Novelty)

| 관점 | 기여 |
|------|------|
| **Inference-time World Model** | 학습 때만 쓰던 V-JEPA2를 추론 루프의 핵심 피드백 엔진으로 복원 |
| **Latent CoT** | 언어 토큰 대신 물리적 잠재 벡터(Δz)를 "사고의 재료"로 활용 |
| **Sequence-level Integration** | Fusion 모듈 없이 QwenVL의 자기 회귀 attention으로 오차 정보를 자연스럽게 통합 |

---

## 7. 파일 구조

```
Recurrent-VLA-JEPA/
├── starVLA/
│   ├── model/
│   │   ├── framework/
│   │   │   └── VLA_JEPA.py              # load_stage3(), _run_qwen_with_correction()
│   │   └── modules/projector/
│   │       └── correction_projector.py  # CorrectionProjector (vj_dim → qwen_dim)
│   └── dataloader/
│       └── stage3_dataset.py            # HDF5에서 이미지 on-the-fly 로드
├── scripts/
│   ├── extract_recurrent_tokens.py      # 오프라인 특징 추출 (Stage 3 메타데이터 포함)
│   ├── train_recurrent_stage3.py        # Stage 3 학습
│   └── run_recurrent_stage3.sh          # 전체 파이프라인 (추출→학습→평가)
└── deployment/model_server/
    └── server_policy.py                 # --stage3_ckpt 인수로 추론 서버 실행
```

---

## 8. 빠른 시작

### 환경 설정

```bash
conda activate vla_jepa
export PYTHONPATH=$(pwd):${PYTHONPATH}
export LIBERO_HOME=/home/choi/LGHA/LIBERO
```

### Stage 3 전체 파이프라인 실행

```bash
# 토큰 추출 → CorrectionProjector 학습 → 평가
bash scripts/run_recurrent_stage3.sh libero_spatial 50
```

### 단계별 실행

```bash
# 1. 오프라인 특징 추출 (Stage 3 메타데이터 포함)
python scripts/extract_recurrent_tokens.py --dataset all

# 2. CorrectionProjector 학습
python -u scripts/train_recurrent_stage3.py \
    --epochs 30 --batch_size 4 --lr 1e-4

# 3. 평가 서버 실행
python deployment/model_server/server_policy.py \
    --ckpt_path <base_ckpt> \
    --stage3_ckpt checkpoints/recurrent_stage3/best.pt \
    --port 15089 --use_bf16
```

---

## 9. 기존 결과 (KF-VLA-JEPA)

| Suite | Baseline | KF (seed 1) | KF (seed 2) | KF (seed 3) | Mean |
|-------|----------|-------------|-------------|-------------|------|
| libero_spatial | — | — | — | — | — |
| libero_10 | 94.0% | 95.6% | — | — | — |

*Stage 3 (Latent CoT) 결과는 학습 완료 후 업데이트 예정*
