import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from MF_AE_Model import MultiFilterMemAE
import os
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MultiFilterTester:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    def load_model(self, model_path):
        """학습된 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 체크포인트에서 모델 설정 정보 가져오기
        use_gabor = checkpoint.get('use_gabor', False)
        
        model = MultiFilterMemAE(
            feature_dim=256,
            mem_dim=512,
            patch_size=7,
            shrink_thres=0.0025,
            use_gabor=use_gabor
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best val loss: {checkpoint.get('val_loss', 'Unknown')}")
        print(f"Using Gabor filters: {use_gabor}")
        
        return model
    
    def process_single_image(self, image_path):
        """단일 이미지 처리"""
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 재구성
            reconstructed = self.model(image_tensor)
            
            # 세부 정보 추출
            details = self.model(image_tensor, return_details=True)
            
            # Loss 계산
            loss, logs = self.model(image_tensor, compute_loss=True)
        
        return {
            'original': image_tensor.squeeze(0).cpu(),
            'reconstructed': reconstructed.squeeze(0).cpu(),
            'loss': loss.item(),
            'logs': logs,
            'details': details,
            'image_path': image_path
        }
    
    def calculate_difference_maps(self, original, reconstructed):
        """차이 맵 계산"""
        # 픽셀 단위 차이 (L1)
        l1_diff = torch.abs(original - reconstructed)
        
        # 픽셀 단위 차이 (L2)
        l2_diff = torch.pow(original - reconstructed, 2)
        
        # 채널별 평균으로 그레이스케일 변환
        l1_gray = torch.mean(l1_diff, dim=0)  # (H, W)
        l2_gray = torch.mean(l2_diff, dim=0)  # (H, W)
        
        return l1_gray, l2_gray
    
    def visualize_results(self, result, save_dir=None):
        """결과 시각화"""
        original = result['original']
        reconstructed = result['reconstructed']
        
        # 차이 맵 계산
        l1_diff, l2_diff = self.calculate_difference_maps(original, reconstructed)
        
        # 시각화 설정
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"MultiFilter MemAE Results\nTotal Loss: {result['loss']:.4f}", fontsize=16)
        
        # 원본 이미지
        axes[0, 0].imshow(original.permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 재구성 이미지
        axes[0, 1].imshow(reconstructed.permute(1, 2, 0).numpy())
        axes[0, 1].set_title('Reconstructed Image')
        axes[0, 1].axis('off')
        
        # L1 차이 히트맵
        im1 = axes[0, 2].imshow(l1_diff.numpy(), cmap='hot', interpolation='nearest')
        axes[0, 2].set_title('L1 Difference Heatmap')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # L2 차이 히트맵
        im2 = axes[1, 0].imshow(l2_diff.numpy(), cmap='hot', interpolation='nearest')
        axes[1, 0].set_title('L2 Difference Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 통계 정보
        axes[1, 1].text(0.1, 0.9, f"MSE Loss: {result['logs']['mse_loss']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.8, f"SSIM Loss: {result['logs'].get('ssim_loss', 0):.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"DCT Loss: {result['logs'].get('dct_loss', 0):.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Entropy: {result['logs']['entropy']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        
        # 브랜치 가중치 표시
        if 'details' in result and 'branch_weights' in result['details']:
            weights = result['details']['branch_weights'].numpy()
            branch_names = ['Original', 'LPF', 'LoG', 'HPF']
            if len(weights) == 5:
                branch_names.append('Gabor')
            
            axes[1, 1].text(0.1, 0.4, "Branch Weights:", transform=axes[1, 1].transAxes, fontsize=12, weight='bold')
            for i, (name, weight) in enumerate(zip(branch_names, weights)):
                axes[1, 1].text(0.1, 0.3 - i*0.05, f"{name}: {weight:.3f}", transform=axes[1, 1].transAxes, fontsize=10)
        
        axes[1, 1].set_title('Statistics')
        axes[1, 1].axis('off')
        
        # 필터별 특징 시각화 (선택적)
        if 'details' in result and 'filtered_results' in result['details']:
            # LPF 결과 표시
            lpf_img = result['details']['filtered_results']['lpf'].squeeze(0).cpu()
            axes[1, 2].imshow(lpf_img.permute(1, 2, 0).numpy())
            axes[1, 2].set_title('LPF Filtered')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 저장
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            image_name = Path(result['image_path']).stem
            save_path = os.path.join(save_dir, f"{image_name}_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def batch_test(self, test_dir, save_dir=None, max_samples=10):
        """배치 테스트"""
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(test_dir).glob(f"*{ext}"))
            image_files.extend(Path(test_dir).glob(f"*{ext.upper()}"))
        
        image_files = image_files[:max_samples]
        print(f"Found {len(image_files)} images for testing")
        
        results = []
        for image_path in image_files:
            print(f"Processing: {image_path}")
            try:
                result = self.process_single_image(str(image_path))
                results.append(result)
                
                # 개별 시각화
                self.visualize_results(result, save_dir)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def compare_normal_vs_abnormal(self, normal_dir, abnormal_dir, save_dir=None):
        """정상 vs 비정상 이미지 비교"""
        # 정상 이미지 처리
        normal_files = list(Path(normal_dir).glob("*.jpg"))[:5]
        abnormal_files = list(Path(abnormal_dir).glob("*.jpg"))[:5] if os.path.exists(abnormal_dir) else []
        
        normal_losses = []
        abnormal_losses = []
        
        print("Processing normal images...")
        for img_path in normal_files:
            result = self.process_single_image(str(img_path))
            normal_losses.append(result['loss'])
        
        if abnormal_files:
            print("Processing abnormal images...")
            for img_path in abnormal_files:
                result = self.process_single_image(str(img_path))
                abnormal_losses.append(result['loss'])
        
        # 손실 분포 시각화
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(normal_losses, bins=10, alpha=0.7, label='Normal', color='blue')
        if abnormal_losses:
            plt.hist(abnormal_losses, bins=10, alpha=0.7, label='Abnormal', color='red')
        plt.xlabel('Reconstruction Loss')
        plt.ylabel('Frequency')
        plt.title('Loss Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        categories = ['Normal']
        values = [np.mean(normal_losses)]
        errors = [np.std(normal_losses)]
        
        if abnormal_losses:
            categories.append('Abnormal')
            values.append(np.mean(abnormal_losses))
            errors.append(np.std(abnormal_losses))
        
        plt.bar(categories, values, yerr=errors, capsize=5, alpha=0.7)
        plt.ylabel('Mean Reconstruction Loss')
        plt.title('Average Loss Comparison')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Normal images - Mean loss: {np.mean(normal_losses):.4f} ± {np.std(normal_losses):.4f}")
        if abnormal_losses:
            print(f"Abnormal images - Mean loss: {np.mean(abnormal_losses):.4f} ± {np.std(abnormal_losses):.4f}")

# 사용 예시
def main():
    # 모델 경로
    model_path = "C:/Users/Admin/Desktop/Python/StegDetector/MultiFilter_AE/checkpoints/multifilter_memae_best.pth"
    
    # 테스터 초기화
    tester = MultiFilterTester(model_path)
    
    # 단일 이미지 테스트
    test_image = "C:/Users/Admin/Desktop/alaska2-image-steganalysis/JUNIWARD/test/63983.jpg"  # 실제 경로로 변경
    if os.path.exists(test_image):
        result = tester.process_single_image(test_image)
        tester.visualize_results(result, save_dir="./results")
    
    # # 배치 테스트
    # test_dir = "C:/Users/Admin/Desktop/alaska2-image-steganalysis/Cover/val"
    # save_dir = "./test_results"
    
    # if os.path.exists(test_dir):
    #     results = tester.batch_test(test_dir, save_dir, max_samples=5)
        
        # # 정상 vs 비정상 비교 (비정상 데이터가 있다면)
        # abnormal_dir = "C:/Users/Admin/Desktop/alaska2-image-steganalysis/JUNIWARD/test/63983.jpg"
        # tester.compare_normal_vs_abnormal(test_dir, abnormal_dir, save_dir)

if __name__ == "__main__":
    main()