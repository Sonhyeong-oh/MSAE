<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>

# Multi-Branch Stacked AutoEncoder for Steganalysis

# 모델 흐름도

<img width="1919" height="624" alt="image" src="https://github.com/user-attachments/assets/dd98c618-7474-483f-b140-35c677be7d7c" />

# 데이터

ALASKA2 (JPG 스테가노그래피 데이터셋)

Cover Image + Stego(J-UNIWARD, J-MiPOD, UERD) Image

Link : https://www.kaggle.com/competitions/alaska2-image-steganalysis

# 파일 안내

## MSCAE_RGBmodel.py - 모델 아키텍처
   
1. MSCAE_RGB_CAE (사전 학습 모델)

- 입력

   RGB 이미지 (3, 256, 256 / 512 * 512 이미지를 resize)

- 아키텍처
  
   다중 브랜치 SRM 필터링 → 90 채널 (30개 필터 X RGB 채널)

   다중 브랜치 인코더 (브랜치별 Stage1 + 공유 Stage2)

   Stage3 = DualAttentionDecoder
   - 어떤 필터가 스테가노그래피 탐지에 중요한지 결정
   - 노이즈 삽입 의심 영역 탐지

   복원을 위한 디코더

   차별적 복원을 위한 채널별 & 공간적 증폭

- 출력
  
   복원된 이미지 + 잠재 표현

2. StegaNetRGB (분류 모델)

- 입력
  
   RGB 이미지 (3, 256, 256)

- 아키텍처
  
   MSCAE_RGB_CAE로부터 인코더/디코더 상속

   공간적 분산 증폭: 높은 분산 영역(가장자리/텍스처) 강조

   하이브리드 분류기: 학습된 특징(2880) + 분산 증폭 오차 특징(90)

- 출력
  
   분류 로짓 [Cover, Stego]

## pretrain_CAERGB.py - 1단계 사전 학습

1. 목적

   SNR Loss + Reconstruction Loss를 사용한 인코더/디코더 사전 학습 데이터셋: BalancedMultiAlgoDataset
   
   Cover + 3개 Stego 알고리즘(JMiPOD, JUNIWARD, UERD) 로드
   
2. 학습 목표
   
- total_loss = snr_weight * snr_loss + recon_weight * recon_loss

- SNR Loss: Stego가 더 복원하기 어렵도록 강제

  목표: stego_error / cover_error >= 1.5

- Recon Loss: Cover의 복원 품질 유지

   주요 특징:
  
   균형잡힌 다중 알고리즘 학습 (J-UNIWARD + J-MiPOD + UERD)
  
   차별적 복원 (Cover: 쉬움, Stego: 어려움)
  
   층화 train/val 분할 (알고리즘 균형 유지)

## MSCAE_trainRGB.py - 2단계 파인튜닝
   
- 목적

   인코더를 고정한 상태에서 분류기 학습

# 시각화

1. SRM Filter

   SRM 필터를 거친 Cover Image와 Stego Image의 픽셀 차이 시각화

<img width="1200" height="1000" alt="difference_residuals" src="https://github.com/user-attachments/assets/f1409d8b-3872-40f0-baf5-c65ebfd60177" />

2. Stage1 출력 시각화

   Multi-Branch Stgae1의 출력 차이 시각화

   <img width="1619" height="715" alt="stage1_JMiPOD_3_00004" src="https://github.com/user-attachments/assets/1fe00d7c-b920-4628-902f-44ee13e158e8" />

2. MSAE 출력 시각화

   MSAE에 의해 재구성된 Cover Image와 Stego Image 차이 시각화 (임시)

<img width="1824" height="913" alt="decoder_JMiPOD_3_00004" src="https://github.com/user-attachments/assets/358e8157-c82c-4f54-a56c-0e3306ec053c" />

# Reference

Shunquan Tan, Bin Li, "Stacked convolutional auto-encoders for steganalysis of digital images", APSIPA 2014
https://ieeexplore.ieee.org/document/7041565
