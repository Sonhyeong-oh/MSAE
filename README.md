# Multi-Branch Stacked AutoEncoder for Steganalysis

## 모델 흐름도

<img width="1919" height="631" alt="image" src="https://github.com/user-attachments/assets/c03d37e6-8a2f-4343-9514-1535f6b71be5" />


파일 안내

## main_pretrain.py
MAE 사전학습(pre-training) 진입점. 인자 파싱 → 데이터셋/분산설정 → 모델 생성 → 옵티마이저/LR 스케줄러 → engine_pretrain.train_one_epoch 호출 루프 형태. 

## engine_pretrain.py
한 에폭의 마스킹+복원 학습 루프. 입력 이미지를 패치 단위로 무작위 마스킹 후, 인코더/디코더를 통해 복원하고 픽셀 L2(또는 유사) 재구성 손실을 계산/로깅합니다. 데이터로더에서 나온 배치를 받아 마스킹 생성 → 전방/역전파 → 미터 집계를 담당합니다. 

## models_mae.py
MAE 모델 정의부.

ViT 인코더(패치 임베딩+클래스 토큰+포지셔널 임베딩)

랜덤 마스킹 로직(토큰을 섞어 keep/drop 인덱스 선택)

작은 디코더(임베딩 차원 축소+별도 포지셔널 임베딩)로 마스킹된 패치 복원

forward에서 (1) 인코더로 가시 토큰만 처리 → (2) 디코더에서 가시+마스크 토큰 합쳐 복원 → (3) 재구성 손실 반환
등의 로직이 들어 있습니다. 

## models_vit.py
기본 ViT 백본(DeiT 기반 수정본). 인코더 블록/멀티헤드 어텐션/MLP 등 표준 구성. MAE의 인코더가 이 구현을 활용합니다. 

## main_finetune.py, engine_finetune.py
사전학습된 MAE 인코더 위에 분류 헤드를 붙여 미세조정(fine-tuning) 하는 스크립트와 루프. ImageNet-1K 등에서의 분류 성능을 재현하는 코드입니다. 

## main_linprobe.py
인코더를 고정(freeze) 하고 선형 분류기만 학습하는 Linear Probe 스크립트. (README/문서에서 안내) 

## util/
분산 학습 유틸, 로깅, 평균계산 메터릭, 체크포인트/시드 고정 등 훈련 보일러플레이트 모음. 

## PRETRAIN.md, FINETUNE.md, README.md
실행 커맨드, 체크포인트 다운로드, 재현 설정 등 사용 가이드. README에는 Colab 데모, 사전학습 가중치 링크, 벤치마크 표가 요약되어 있습니다. 

## submitit_*.py
클러스터 환경에서 Submitit으로 잡 제출을 돕는 스크립트. (대규모 분산 학습 배치용)
