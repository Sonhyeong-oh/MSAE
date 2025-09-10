from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import models_mae_pp
import numpy as np
import torch
from tqdm import tqdm

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# 내 이미지 데이터셋 (예: 폴더 구조가 ImageFolder일 때)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
dataset = datasets.ImageFolder("C:/Users/Admin/Desktop/Image_data/Original/", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 0) device 고정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 1) 모델 로드 → 디바이스로 이동 → (그 다음 옵티마이저 생성)
model = models_mae_pp.mae_vit_large_patch16_dec512d8b(
    norm_pix_loss=False,
    use_perc=True, use_edge=True,
    lambda_perc=0.01, lambda_edge=10.0
)
# ckpt = torch.load(r"C:/Users/Admin/Desktop/Python/StegDetector/mae-main/mae-main/mae_visualize_vit_large.pth",
#                   map_location='cpu')
# model.load_state_dict(ckpt['model'], strict=False)
model = model.to(device)                     # ✅ 모델을 GPU/CPU로 이동
model.train()

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # ✅ tqdm으로 DataLoader 감싸기
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True, dtype=torch.float32)

        loss, pred, mask = model(imgs, mask_ratio=0.75)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        # ✅ 진행 중 평균 손실 업데이트
        avg_loss = running_loss / (len(pbar) * loader.batch_size)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    epoch_loss = running_loss / len(dataset)
    print(f"[{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")

# ✅ 학습 완료 후 저장
save_path = "C:/Users/Admin/Desktop/Python/StegDetector/mae-main/mae-main/new_mae_vit.pth"
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': num_epochs,
}, save_path)

print(f"새로운 MAE 모델 저장 완료: {save_path}")
