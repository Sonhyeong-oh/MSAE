# Multi-Branch Stacked AutoEncoder for Steganalysis

## 모델 흐름도

<img width="1919" height="624" alt="image" src="https://github.com/user-attachments/assets/dd98c618-7474-483f-b140-35c677be7d7c" />


## 파일 안내

1. MSCAE_RGBmodel.py - 모델 아키텍처
   
- 목적

모든 신경망 아키텍처 정의

## MSCAE_RGB_CAE (사전 학습 모델)

- 입력
  RGB 이미지 (3, 256, 256)

- 아키텍처
  
다중 브랜치 SRM 필터링 → 90 채널 (4개 브랜치 × 22.5개 필터)
다중 브랜치 인코더 (브랜치별 Stage1 + 공유 Stage2/3)
복원을 위한 디코더
차별적 복원을 위한 채널별 & 공간적 증폭

- 출력
  
복원된 이미지 + 잠재 표현

## StegaNetRGB (분류 모델)

- 입력
  
RGB 이미지 (3, 256, 256)

- 아키텍처
  
MSCAE_RGB_CAE로부터 인코더/디코더 상속
공간적 분산 증폭: 높은 분산 영역(가장자리/텍스처) 강조
하이브리드 분류기: 학습된 특징(2880) + 분산 증폭 오차 특징(90)

- 출력
  
분류 로짓 [Cover, Stego]

- SNRLoss (커스텀 손실 함수)
  
목적: 차별적 복원 강제 (Stego가 Cover보다 복원하기 어렵도록)
수식: Loss = max(0, target_ratio - stego_error/cover_error)²

- Recon Loss

Cover의 복원 품질 유지
주요 특징:
균형잡힌 다중 알고리즘 학습 (알고리즘별 과적합 방지)
차별적 복원 (Cover: 쉬움, Stego: 어려움)
층화 train/val 분할 (알고리즘 균형 유지)
사용법:
python pretrain_RGBCAE.py \
    --cover_dir /path/to/covers \
    --jmipod_dir /path/to/JMiPOD \
    --juniward_dir /path/to/JUNIWARD \
    --uerd_dir /path/to/UERD \
    --snr_weight 30.0 \
    --recon_weight 3.0 \
    --epochs 50
출력: checkpoints_cae/cae_rgb_balanced_stage1.pth
3. MSCAE_trainRGB.py - 2단계 파인튜닝
목적: 인코더를 고정한 상태에서 분류기 학습 (전이 학습) 데이터셋: 사전 학습과 동일한 균형잡힌 다중 알고리즘 데이터셋 학습 전략:
# 인코더/디코더 고정 (사전 학습된 지식)
for param in model.encoder.parameters():
    param.requires_grad = False

# 분류기만 학습
optimizer = AdamW(classifier_params, lr=1e-4)
모델 구성:
전체 파라미터: ~30M
├─ 고정됨 (인코더/디코더): ~28.5M (95%)
└─ 학습 가능 (분류기): ~1.5M (5%)
손실 함수:
# 분류 손실만 사용 (복원 손실 없음)
loss = CrossEntropy(logits, labels)
주요 특징:
전이 학습 (사전 학습된 인코더 활용)
균형잡힌 다중 알고리즘 검증
실시간 복원 비율 모니터링 (Cover vs Stego)
강건한 오차 특징을 위한 공간적 분산 증폭
