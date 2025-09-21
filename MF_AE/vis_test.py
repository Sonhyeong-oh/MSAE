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
    
    def calculate_difference_maps(self, original, reconstructed):
        """차이 맵 계산"""
        l1_diff = torch.abs(original - reconstructed)
        l2_diff = torch.pow(original - reconstructed, 2)
        
        l1_gray = torch.mean(l1_diff, dim=0)
        l2_gray = torch.mean(l2_diff, dim=0)
        
        return l1_gray, l2_gray
    
    def visualize_filter_responses(self, result, save_dir=None):
        """필터 응답 시각화"""
        if 'details' not in result or 'filtered_results' not in result['details']:
            print("Filtered results not available for visualization.")
            return
            
        filtered_results = result['details']['filtered_results']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Filter Responses", fontsize=16)
        
        # Original
        axes[0, 0].imshow(filtered_results['original'].squeeze(0).cpu().permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Hybrid Laplacian
        hybrid_img = filtered_results['hybrid'].squeeze(0).cpu()
        axes[0, 1].imshow(hybrid_img.permute(1, 2, 0).numpy())
        axes[0, 1].set_title('Hybrid Laplacian (Edges)')
        axes[0, 1].axis('off')
        
        # DCT (16개 채널을 1개로 평균)
        dct_img_16_ch = filtered_results['dct'].squeeze(0).cpu()
        dct_img = dct_img_16_ch.mean(dim=0).numpy()
        im = axes[0, 2].imshow(dct_img, cmap='hot', interpolation='nearest')
        axes[0, 2].set_title('DCT (Frequency)')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 어텐션 맵 시각화
        if 'attention_maps' in result['details']:
            att_maps = result['details']['attention_maps']
            if 'unified_attention' in att_maps:
                unified_att = att_maps['unified_attention'].squeeze(0).cpu()[0].numpy()
                scale_factor = filtered_results['original'].shape[2] // unified_att.shape[0]
                unified_att_resized = np.repeat(np.repeat(unified_att, scale_factor, axis=0), scale_factor, axis=1)
                
                im = axes[1, 0].imshow(unified_att_resized, cmap='hot', alpha=0.7)
                axes[1, 0].set_title('Unified Attention')
                axes[1, 0].axis('off')
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Edge Attention (Unified Attention과 동일한 맵이므로 생략 가능)
        axes[1, 1].set_title('Reserved')
        axes[1, 1].axis('off')
        
        # 통계 정보
        stats_text = f"Filter Response Statistics:\n"
        stats_text += f"Hybrid Max: {result['logs'].get('hybrid_max', 0):.3f}\n"
        stats_text += f"Hybrid Mean: {result['logs'].get('hybrid_mean', 0):.3f}\n"
        stats_text += f"DCT Max: {result['logs'].get('dct_max', 0):.3f}\n"
        stats_text += f"DCT Mean: {result['logs'].get('dct_mean', 0):.3f}\n\n"
        
        stats_text += f"Dynamic Thresholds:\n"
        stats_text += f"Hybrid: {result['logs'].get('hybrid_threshold', 0):.3f}\n"
        stats_text += f"DCT: {result['logs'].get('dct_threshold', 0):.3f}"
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Filter Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            image_name = Path(result['image_path']).stem
            save_path = os.path.join(save_dir, f"{image_name}_filters.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Filter visualization saved to: {save_path}")
        
        plt.show()
        return fig
    
    def visualize_results(self, result, save_dir=None):
        """결과 시각화"""
        original = result['original']
        reconstructed = result['reconstructed']
        
        l1_diff, l2_diff = self.calculate_difference_maps(original, reconstructed)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"MultiFilter AutoEncoder Results\nTotal Loss: {result['loss']:.4f}", fontsize=16)
        
        axes[0, 0].imshow(original.permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(reconstructed.permute(1, 2, 0).numpy())
        axes[0, 1].set_title('Reconstructed Image')
        axes[0, 1].axis('off')
        
        im1 = axes[0, 2].imshow(l1_diff.numpy(), cmap='hot', interpolation='nearest')
        axes[0, 2].set_title('L1 Difference Heatmap')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        im2 = axes[1, 0].imshow(l2_diff.numpy(), cmap='hot', interpolation='nearest')
        axes[1, 0].set_title('L2 Difference Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 통계 정보 (새로운 loss 구조에 맞게 수정)
        stats_text = f"Loss Components:\n"
        stats_text += f"Base MSE: {result['logs'].get('base_mse', 0):.4f}\n"
        stats_text += f"Hybrid Loss: {result['logs'].get('hybrid', 0):.4f}\n"
        stats_text += f"DCT Loss: {result['logs'].get('dct', 0):.4f}\n"
        stats_text += f"Total Loss: {result['logs'].get('total', 0):.4f}\n\n"
        
        stats_text += f"Specialization:\n"
        stats_text += f"Original: {result['logs'].get('original_spec', 0):.4f}\n"
        stats_text += f"Hybrid: {result['logs'].get('hybrid_spec', 0):.4f}\n"
        stats_text += f"DCT: {result['logs'].get('dct_spec', 0):.4f}\n\n"
        
        stats_text += f"Dynamic Thresholds:\n"
        stats_text += f"Hybrid: {result['logs'].get('hybrid_threshold', 0):.3f}\n"
        stats_text += f"DCT: {result['logs'].get('dct_threshold', 0):.3f}"
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Loss Statistics')
        axes[1, 1].axis('off')
        
        # DCT Filtered Input 표시
        if 'details' in result and 'filtered_results' in result['details']:
            dct_img_16_ch = result['details']['filtered_results']['dct'].squeeze(0).cpu()
            dct_img = dct_img_16_ch.mean(dim=0).numpy()
            im3 = axes[1, 2].imshow(dct_img, cmap='hot', interpolation='nearest')
            axes[1, 2].set_title('DCT Filtered Input (Mean)')
            axes[1, 2].axis('off')
            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            image_name = Path(result['image_path']).stem
            save_path = os.path.join(save_dir, f"{image_name}_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def analyze_branch_performance(self, result):
        """브랜치별 성능 분석"""
        if 'details' not in result or not isinstance(result['details'], dict):
            print("Details not available or invalid format")
            return
            
        print("\n=== Branch Performance Analysis ===")
        
        details = result['details']
        if 'branch_features' in details and isinstance(details['branch_features'], (list, tuple)):
            branch_features = details['branch_features']
            print(f"Branch feature shapes:")
            for i, feat in enumerate(branch_features):
                branch_names = ['Original', 'Hybrid', 'DCT']
                if i < len(branch_names) and hasattr(feat, 'shape'):
                    print(f"  {branch_names[i]}: {feat.shape}")
        
        if 'attention_maps' in details and isinstance(details['attention_maps'], dict):
            att_maps = details['attention_maps']
            print(f"\nAttention Statistics:")
            
            if 'unified_attention' in att_maps and hasattr(att_maps['unified_attention'], 'squeeze'):
                unified_att = att_maps['unified_attention'].squeeze(0).cpu()
                print(f"  Unified Attention - Mean: {unified_att.mean():.4f}, Std: {unified_att.std():.4f}")
            
            if 'hybrid_attention' in att_maps and hasattr(att_maps['hybrid_attention'], 'squeeze'):
                hybrid_att = att_maps['hybrid_attention'].squeeze(0).cpu()
                print(f"  Hybrid Attention - Mean: {hybrid_att.mean():.4f}, Std: {hybrid_att.std():.4f}")
                
            if 'dct_attention' in att_maps and hasattr(att_maps['dct_attention'], 'squeeze'):
                dct_att = att_maps['dct_attention'].squeeze(0).cpu()
                print(f"  DCT Attention - Mean: {dct_att.mean():.4f}, Std: {dct_att.std():.4f}")

        print(f"\nFilter Response Analysis:")
        print(f"  Hybrid Max/Mean: {result['logs'].get('hybrid_max', 0):.3f} / {result['logs'].get('hybrid_mean', 0):.3f}")
        print(f"  DCT Max/Mean: {result['logs'].get('dct_max', 0):.3f} / {result['logs'].get('dct_mean', 0):.3f}")
        print(f"  Dynamic Hybrid Threshold: {result['logs'].get('hybrid_threshold', 0):.3f}")
        print(f"  Dynamic DCT Threshold: {result['logs'].get('dct_threshold', 0):.3f}")
    
    def batch_test(self, test_dir, save_dir=None, max_samples=10):
        """배치 테스트"""
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
                
                self.analyze_branch_performance(result)
                
                self.visualize_results(result, save_dir)
                self.visualize_filter_responses(result, save_dir)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return results
    
    def compare_normal_vs_abnormal(self, normal_dir, abnormal_dir, save_dir=None):
        """정상 vs 비정상 이미지 비교"""
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

def main():
    model_path = "C:/Users/Admin/Desktop/Python/StegDetector/MultiFilter_AE/checkpoints/multifilter_ae_last.pth"
    
    tester = MultiFilterTester(model_path)
    
    test_image = "C:/Users/Admin/Desktop/alaska2-image-steganalysis/UERD/test/63987.jpg"
    if os.path.exists(test_image):
        result = tester.process_single_image(test_image)
        tester.analyze_branch_performance(result)
        tester.visualize_results(result, save_dir="./results")
        tester.visualize_filter_responses(result, save_dir="./results")
    
    # # 배치 테스트
    # test_dir = "C:/Users/Admin/Desktop/alaska2-image-steganalysis/Cover/val"
    # save_dir = "./test_results"
    
    # if os.path.exists(test_dir):
    #     results = tester.batch_test(test_dir, save_dir, max_samples=5)
        
    #     abnormal_dir = "C:/Users/Admin/Desktop/alaska2-image-steganalysis/JUNIWARD/test"
    #     tester.compare_normal_vs_abnormal(test_dir, abnormal_dir, save_dir)

if __name__ == "__main__":
    main()