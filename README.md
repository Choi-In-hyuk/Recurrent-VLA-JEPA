# Recurrent VLA-JEPA

> **V-JEPA2 월드 모델 예측기를 추론 루프에 통합하여 시간적 일관성을 가진 로봇 제어를 실현하는 VLA 프레임워크**

이 레포지토리는 [VLA-JEPA](https://arxiv.org/abs/2602.10098) (Sun et al., 2026)를 기반으로 두 가지 핵심 개선을 구현합니다:

1. **KF-VLA-JEPA** — 추론 시간에 Learned Kalman Filter로 `embodied_action_tokens`를 시간적으로 평활화
2. **Recurrent-JEPA** — V-JEPA2 예측기를 추론 루프에 통합하여 월드 모델과 정책의 일관성 확보

---

## 배경: 해결하려는 두 가지 문제

### 문제 1 — 학습/추론 불일치 (Train-Test Inconsistency)

VLA-JEPA는 학습 시 V-JEPA2 예측기(world model)를 사용해 미래 시각 토큰을 예측하고 이를 정책 학습에 활용한다. 그러나 **추론 시에는 예측기가 완전히 무시**되고, QwenVL이 현재 프레임만 처리한다.

```
[학습]   vj_encoder → vj_predictor → DiT
[추론]   vj_encoder              → DiT   ← predictor가 누락됨
```

### 문제 2 — Stateless 시각 인코딩

QwenVL은 각 프레임을 독립적으로 인코딩한다. 장면이 거의 바뀌지 않아도 `embodied_action_tokens`가 타임스텝마다 급격히 변동하여 DiT 액션 헤드에 불일치가 전파된다.

---

## 아키텍처

### VLA-JEPA 기본 구조

```
이미지 I_t (agentview + wrist) ─────────────────────────────────┐
언어 명령 l ─────────────────────────────────────────────────────┤
                                                                  ↓
                                              ┌─────────────────────────────┐
                                              │   QwenVL (Qwen3-VL-2B)      │
                                              │   - 비전 + 언어 공동 인코딩  │
                                              └──────────┬──────────────────┘
                                                         │
                              ┌──────────────────────────┼──────────────────────┐
                              ↓                          ↓                      ↓
                    action_tokens              embodied_action_tokens      hidden states
                  (T-1)×8×2048               32×2048                    (KF/EMA 용도)
                      │
                      ↓
이미지 I_t ──→ vj_encoder ──→ video_embeddings (T×256, 2048)
                      │              │
                      └──────────────┘
                              ↓
                       vj_predictor
                (action-conditioned 미래 예측)
                              │
                              ↓
                    predicted_states (T×256, 2048)
```

```
embodied_action_tokens (32, 2048)
              ↓
         DiT (Flow-matching)
              │
          state s_t (8,)
              ↓
     action chunk â_t (7, action_dim)
```

### Recurrent-JEPA 확장 구조

```
타임스텝 t

z_{t-1} (이전 vj_obs)
a_{t-1} (이전 action_tokens)     ←── 이전 스텝에서 저장
           │
           ↓
    vj_predictor (frozen)
           │
           ↓
    ẑ_t : 예측된 시각 토큰 (256, 2048)

이미지 I_t ──→ vj_encoder ──→ z_t : 현재 관측 토큰 (256, 2048)

           ┌─────────────────────┐
z_t ──────→│                     │
           │  LearnedGatingFusion│──→ f_t (256, 2048)
ẑ_t ──────→│                     │
           └─────────────────────┘
                       │
                       ↓
              VJtoDiTProjection
                       │
                       ↓
              c_t : 조건 토큰 (8, 2048)

embodied_action_tokens e_t (32, 2048)
              │
              ├── concat ──→ DiT 입력 x_t (40, 2048)
              │
         c_t ─┘

              ↓
         DiT (Flow-matching)
              ↓
     action chunk â_t (7, 7)
```

---

## 수식

### 1. V-JEPA2 인코딩

현재 프레임을 공간 토큰으로 인코딩:

$$z_t = \text{Enc}_\theta(I_t) \in \mathbb{R}^{N \times D}$$

- $N = 256$ : 프레임당 공간 토큰 수
- $D = 2048$ : 토큰 임베딩 차원 (ViT-L 1024 × 2 views)

### 2. 액션 조건 예측

이전 관측과 액션 토큰으로 현재 프레임을 예측:

$$\hat{z}_t = \text{Pred}_\phi(z_{t-1},\ a^{\text{action}}_{t-1}) \in \mathbb{R}^{N \times D}$$

- $a^{\text{action}}_{t-1}$ : QwenVL이 출력한 `<|action_i|>` 숨겨진 상태 $\in \mathbb{R}^{(T-1) \times 8 \times 2048}$
- 예측기는 학습 완료 후 **frozen**

Cold start (첫 스텝): $\hat{z}_0 = z_0$ (예측 없이 관측만 사용)

### 3. 학습된 게이팅 융합 (LearnedGatingFusion)

관측 토큰과 예측 토큰을 cosine 유사도를 추가 피처로 활용하여 동적으로 융합:

$$\text{cos}_t = \text{CosineSim}(z_t,\ \hat{z}_t) \in \mathbb{R}^{N \times 1}$$

$$\alpha_t = \sigma\!\left(W_g \cdot [z_t \,;\, \hat{z}_t \,;\, \text{cos}_t]\right) \in \mathbb{R}^{N \times D}$$

$$f_t = \alpha_t \odot z_t + (1 - \alpha_t) \odot \hat{z}_t \in \mathbb{R}^{N \times D}$$

- $W_g \in \mathbb{R}^{D \times (2D+1)}$ : 학습 가능한 게이팅 행렬
- $\alpha_t \approx 1$ : 예측이 부정확할 때 관측을 더 신뢰
- $\alpha_t \approx 0$ : 예측이 정확할 때 예측 토큰 활용

### 4. VJtoDiT 프로젝션 (VJtoDiTProjection)

256개 공간 토큰을 DiT 조건 입력으로 압축:

$$f_t^{\text{pool}} = \text{GroupMeanPool}(f_t,\ n_{\text{cond}}) \in \mathbb{R}^{n_{\text{cond}} \times D}$$

$$c_t = \text{LayerNorm}(W_p \cdot f_t^{\text{pool}}) \in \mathbb{R}^{n_{\text{cond}} \times D}$$

- $n_{\text{cond}} = 8$ : 조건 토큰 수 (256 → 8로 압축, 그룹 크기 32)
- $W_p \in \mathbb{R}^{D \times D}$ : 학습 가능한 프로젝션 행렬

### 5. DiT 조건부 입력

```
x_t = [e_t ; c_t]  ∈  R^{40 × 2048}
      └32 tok┘└8 tok┘
```

$$x_t = \left[e_t^{(1:32)} \,;\, c_t^{(1:8)}\right] \in \mathbb{R}^{40 \times 2048}$$

- $e_t$ : QwenVL `embodied_action_tokens` (언어+시각 조건)
- $c_t$ : Recurrent JEPA 월드 모델 조건 (시간적 예측 정보)

### 6. Flow-Matching 액션 손실

$$\psi_t = (1-t)\,\varepsilon + t\,a, \quad \varepsilon \sim \mathcal{N}(0, I), \quad t \sim \text{Beta}(1.5,\ 1.0)$$

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\,\varepsilon,\,a}\!\left[\left\|v_\theta(\psi_t,\,t,\,x_t,\,s_t) - (a - \varepsilon)\right\|^2\right]$$

- $v_\theta$ : DiT 속도 예측기
- $s_t \in \mathbb{R}^8$ : 로봇 proprioceptive 상태 (joint 7 + gripper 1)
- 학습 시 동일 샘플을 `repeated_diffusion_steps=8`회 반복하여 손실 계산

---

## KF-VLA-JEPA: Kalman Filter 평활화

Recurrent-JEPA와 독립적으로 적용 가능한 추론 시간 개선 방법.

### 오프라인 학습 (1회)

1. 데모 데이터에서 `embodied_action_tokens` 추출 → $(T, 2048)$
2. PCA 인코더 $E \in \mathbb{R}^{64 \times 2048}$ 학습 (Truncated SVD)
3. AR(1) 전이 행렬 $A \in \mathbb{R}^{64 \times 64}$ 학습 (Least Squares)

### 온라인 추론 (매 스텝)

**예측 단계:**

$$\hat{z}_t^- = A z_{t-1}, \quad P_t^- = A P_{t-1} A^\top + Q$$

**업데이트 단계:**

$$K_t = P_t^-\!\left(P_t^- + R\right)^{-1}$$

$$z_t = \hat{z}_t^- + K_t\!\left(z_t^{\text{obs}} - \hat{z}_t^-\right), \quad P_t = (I - K_t)\,P_t^-$$

필터링된 잠재 $z_t$를 디코딩하여 32개 `embodied_action_tokens` 전체에 잔차 보정 적용.

---

## 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `vj_dim` | 2048 | V-JEPA2 공간 토큰 차원 (1024 × 2 views) |
| `dit_dim` | 2048 | DiT 조건 입력 차원 |
| `spatial_tokens` | 256 | 프레임당 공간 토큰 수 |
| `n_cond_tokens` | 8 | DiT에 전달되는 압축 조건 토큰 수 |
| `embodied_action_tokens` | 32 | QwenVL 언어-시각 조건 토큰 수 |
| `action_dim` | 7 | delta_qpos 액션 차원 |
| `state_dim` | 8 | 로봇 상태 차원 (joint 7 + gripper 1) |
| `action_horizon` | 7 | DiT가 예측하는 미래 액션 스텝 수 |
| `num_frames` | 8 | vj_encoder 입력 프레임 수 |
| `repeated_diffusion_steps` | 8 | 학습 시 손실 반복 횟수 |
| `num_inference_timesteps` | 4 | DiT 추론 시 디노이징 스텝 수 |
| `KF latent_dim` | 64 | Kalman Filter 잠재 차원 |

---

## 학습 설정

### Phase 2: Recurrent-JEPA 융합 모듈 학습

| 모듈 | 학습 여부 | 학습률 |
|---|---|---|
| QwenVL | ❌ Frozen | — |
| vj_encoder | ❌ Frozen | — |
| vj_predictor | ❌ Frozen | — |
| **LearnedGatingFusion** | ✅ 학습 (from scratch) | 1e-4 |
| **VJtoDiTProjection** | ✅ 학습 (from scratch) | 1e-4 |
| **DiT (action_model)** | ✅ Fine-tune | 1e-4 |

**오프라인 특징 추출 기반 학습 방식:**

직접 HDF5 데모 → V-JEPA 인코더 + QwenVL로 `(vj_obs, vj_pred, emb_tokens, actions, states)` 사전 추출 → 가벼운 융합 모듈만 학습 (GPU 1장, 수 시간 내 완료)

---

## 실험 결과

### KF-VLA-JEPA (LIBERO 벤치마크)

| Suite | Baseline | KF (ours) | Δ | p-value |
|---|---|---|---|---|
| LIBERO-Spatial | 95.20% ± 0.49% | **97.13% ± 0.34%** | **+1.93%p** | **0.010\*** |
| LIBERO-Object | 99.93% ± 0.09% | 99.87% ± 0.09% | −0.07%p | 0.519 (천장 효과) |
| LIBERO-Goal | 97.07% ± 0.90% | **97.73% ± 0.19%** | +0.67%p | 0.363 |

- LIBERO-Spatial에서 통계적으로 유의미한 성능 향상 (p=0.010)
- 분산 감소: LIBERO-Goal std 0.90% → 0.19%

---

## 설치

```bash
git clone https://github.com/Choi-In-hyuk/Recurrent-VLA-JEPA
cd Recurrent-VLA-JEPA

conda create -n vla_jepa python=3.10 -y
conda activate vla_jepa

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

**LIBERO 환경** (별도 conda env):
```bash
# 공식 설치 가이드: https://github.com/Lifelong-Robot-Learning/LIBERO
```

---

## 사용법

### Phase 1: Recurrent 추론 sanity check (cosine similarity 로깅)

```bash
bash examples/LIBERO/eval_libero_recurrent.sh libero_spatial 50
```

→ 서버 로그에서 `pred/obs cosine_sim ≈ 0.45` 확인 (예측기가 의미 있는 예측을 하고 있음)

### Phase 2: 전체 파이프라인 (특징 추출 → 학습 → 평가)

```bash
bash scripts/run_recurrent_phase2.sh libero_spatial 50
```

내부 실행 순서:
1. `scripts/extract_recurrent_tokens.py` — LIBERO HDF5에서 `(vj_obs, vj_pred, emb_tokens, actions, states)` 추출
2. `scripts/train_recurrent_fusion.py` — 융합 모듈 학습, `checkpoints/recurrent_jepa_ft/best.pt` 저장
3. `eval_libero.py` — Phase 2 체크포인트로 LIBERO 평가

### KF 파이프라인

```bash
bash run_all_suites.sh         # 전체 LIBERO 스위트
bash run_libero_pro.sh         # LIBERO-PRO 교란 평가
```

---

## 서버 실행 플래그

| 플래그 | 설명 |
|---|---|
| `--recurrent` | Phase 1: Recurrent 추론 (cosine_sim 로깅만) |
| `--recurrent_ckpt <path>` | Phase 2: 학습된 융합 모듈 체크포인트 로드 |
| `--lds_path <path>` | KF 필터 활성화 (LDS `.npz` 파일) |
| `--kf_q` | KF 프로세스 노이즈 (기본값: 0.1) |
| `--kf_r` | KF 관측 노이즈 (기본값: 5.0) |
| `--ema_alpha` | EMA 평활화 계수 (KF와 상호 배타적) |

---

## 디렉토리 구조

```
Recurrent-VLA-JEPA/
├── starVLA/
│   ├── model/
│   │   ├── framework/
│   │   │   └── VLA_JEPA.py              # 메인 모델 (Recurrent 루프 포함)
│   │   └── modules/
│   │       ├── fusion/
│   │       │   └── gating.py            # LearnedGatingFusion
│   │       ├── projector/
│   │       │   └── vj_to_dit.py         # VJtoDiTProjection
│   │       ├── action_model/
│   │       │   └── GR00T_ActionHeader.py # DiT flow-matching 헤드
│   │       └── world_model/
│   │           └── vj2_predictor.py      # V-JEPA2 예측기
│   └── dataloader/
│       └── recurrent_token_dataset.py   # 오프라인 토큰 Dataset
├── scripts/
│   ├── extract_recurrent_tokens.py      # 특징 추출 스크립트
│   ├── train_recurrent_fusion.py        # Phase 2 학습 스크립트
│   ├── run_recurrent_phase2.sh          # 전체 파이프라인
│   └── config/
│       └── recurrent_jepa_ft.yaml       # 학습 설정
├── deployment/
│   └── model_server/
│       └── server_policy.py             # WebSocket 정책 서버
├── examples/LIBERO/
│   ├── eval_libero.py                   # LIBERO 평가
│   └── eval_libero_recurrent.sh         # Phase 1 평가 스크립트
└── results/
    └── results.md                       # 실험 결과 및 분석
```

---

## 참고 문헌

- **VLA-JEPA**: Sun et al., "VLA-JEPA: A World Model for Vision-Language-Action Policies", 2026
- **V-JEPA2**: Meta AI, "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning", 2025
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
- **LIBERO**: Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", NeurIPS 2023
