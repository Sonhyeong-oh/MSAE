import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from MFwM_AE import MultiFilterAutoEncoder
import os
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MultiFilterTester:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def load_model(self, model_path):
        """학습된 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = MultiFilterAutoEncoder(feature_dim=256).to(self.device)
        
        if any(key.startswith('module.') for key in checkpoint['model_state_dict'].keys()):
            state_dict = {key.replace('module.', ''): value for key, value in checkpoint['model_state_dict'].items()}
        else:
            state_dict = checkpoint['model_state_dict']
            
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best val loss: {checkpoint.get('val_loss', 'Unknown')}")
        
        return model
    
    def process_single_image(self, image_path):
        """단일 이미지 처리"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 재구성된 이미지와 상세 정보를 한 번에 가져옵니다.
            reconstructed, logs = self.model(image_tensor, compute_loss=True, include_specialization_loss=True)
            details = self.model(image_tensor, return_dcts=True)
        
        # reconstruct, logs는 (tensor, dict) 이므로, reconstructed는 tensor
        # details는 dict이므로, 'reconstructed' 키를 통해 텐서를 가져올 수 있습니다.
        reconstructed_image = details['reconstructed']
        
        return {
            'original': image_tensor.squeeze(0).cpu(),
            'reconstructed': reconstructed_image.squeeze(0).cpu(),
            'loss': logs['total'],
            'logs': logs,
            'details': details,
            'image_path': image_path
        }


def main():
    model_path = "C:/Users/Admin/Desktop/pth/resnet-resnet(25.09.20)/multifilter_ae_last.pth"
    tester = MultiFilterTester(model_path)
    
    # 분석을 위한 4개의 데이터셋 경로
    dataset_paths = {
        "Cover": "C:/Users/Admin/Desktop/alaska2-image-steganalysis/Cover/val",
        "JUNIWARD": "C:/Users/Admin/Desktop/alaska2-image-steganalysis/JUNIWARD/test",
        "UERD": "C:/Users/Admin/Desktop/alaska2-image-steganalysis/UERD/test",
        "JMiPOD": "C:/Users/Admin/Desktop/alaska2-image-steganalysis/JMiPOD/test"
    }

    all_results = {}
    
    for dataset_name, path in dataset_paths.items():
        if os.path.exists(path):
            print(f"\n===== Analyzing {dataset_name} dataset =====")
            # 배치 테스트 실행 (시각화 제외)
            results = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(path).glob(f"*{ext}"))
                image_files.extend(Path(path).glob(f"*{ext.upper()}"))
            
            image_files = image_files[:100]  # 각 데이터셋에서 최대 100개 이미지 사용
            
            for image_path in image_files:
                try:
                    result = tester.process_single_image(str(image_path))
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            
            all_results[dataset_name] = results
            
            # 평균 지표 계산 및 출력
            if results:
                # 지표들을 저장할 리스트
                unified_att_means, hybrid_att_means, dct_att_means = [], [], []
                hybrid_maxes, hybrid_means, dct_maxes, dct_means = [], [], [], []
                base_mses, hybrid_losses, dct_losses, total_losses = [], [], [], []

                for res in results:
                    logs = res['logs']
                    details = res['details']
                    
                    # Attention Statistics
                    if 'attention_maps' in details:
                        att_maps = details['attention_maps']
                        if 'unified_attention' in att_maps:
                            unified_att_means.append(att_maps['unified_attention'].mean().item())
                        if 'hybrid_attention' in att_maps:
                            hybrid_att_means.append(att_maps['hybrid_attention'].mean().item())
                        if 'dct_attention' in att_maps:
                            dct_att_means.append(att_maps['dct_attention'].mean().item())
                    
                    # Filter Response Analysis
                    if 'hybrid_max' in logs:
                        hybrid_maxes.append(logs['hybrid_max'])
                    if 'hybrid_mean' in logs:
                        hybrid_means.append(logs['hybrid_mean'])
                    if 'dct_max' in logs:
                        dct_maxes.append(logs['dct_max'])
                    if 'dct_mean' in logs:
                        dct_means.append(logs['dct_mean'])
                    
                    # Loss Components
                    if 'base_mse' in logs:
                        base_mses.append(logs['base_mse'])
                    if 'hybrid' in logs:
                        hybrid_losses.append(logs['hybrid'])
                    if 'dct' in logs:
                        dct_losses.append(logs['dct'])
                    if 'total' in logs:
                        total_losses.append(logs['total'])

                # 평균값 출력
                print(f"\nAverage Metrics for {dataset_name}:")
                
                print("  Attention Statistics:")
                print(f"    Unified Attention - Mean: {np.mean(unified_att_means):.4f}")
                print(f"    Hybrid Attention - Mean: {np.mean(hybrid_att_means):.4f}")
                print(f"    DCT Attention - Mean: {np.mean(dct_att_means):.4f}")
                
                print("\n  Filter Response Analysis:")
                print(f"    Hybrid Max/Mean: {np.mean(hybrid_maxes):.3f} / {np.mean(hybrid_means):.3f}")
                print(f"    DCT Max/Mean: {np.mean(dct_maxes):.3f} / {np.mean(dct_means):.3f}")
                
                print("\n  Loss Components:")
                print(f"    Base MSE: {np.mean(base_mses):.4f}")
                print(f"    Hybrid Loss: {np.mean(hybrid_losses):.4f}")
                print(f"    DCT Loss: {np.mean(dct_losses):.4f}")
                print(f"    Total Loss: {np.mean(total_losses):.4f}")
            else:
                print(f"No images processed for {dataset_name}.")
        else:
            print(f"Dataset path not found: {path}")

if __name__ == "__main__":
    main()