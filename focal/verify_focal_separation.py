"""
FOCAL의 Shared/Specific Vector 분리 검증 스크립트
"""
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def verify_orthogonality(shared_features, private_features):
    """
    방법 1: Orthogonality 확인
    Shared와 Private vector가 직교(orthogonal)하는지 확인
    """
    print("\n" + "="*60)
    print("방법 1: Orthogonality 검증")
    print("="*60)
    
    # Cosine similarity 계산
    shared_norm = shared_features / (torch.norm(shared_features, dim=-1, keepdim=True) + 1e-8)
    private_norm = private_features / (torch.norm(private_features, dim=-1, keepdim=True) + 1e-8)
    
    # Inner product (직교하면 0에 가까워야 함)
    cosine_sim = (shared_norm * private_norm).sum(dim=-1)
    
    mean_sim = cosine_sim.abs().mean().item()
    print(f"평균 Cosine Similarity: {mean_sim:.4f}")
    print(f"판정: {'✓ 잘 분리됨' if mean_sim < 0.1 else '✗ 분리 부족'}")
    print(f"     (0에 가까울수록 직교 = 잘 분리됨)")
    
    return mean_sim


def verify_cross_modal_similarity(mod_features):
    """
    방법 2: Cross-modal Similarity 확인
    Shared space에서는 모달리티 간 유사도가 높아야 함
    Private space에서는 모달리티 간 유사도가 낮아야 함
    """
    print("\n" + "="*60)
    print("방법 2: Cross-modal Similarity 검증")
    print("="*60)
    
    modalities = list(mod_features.keys())
    
    # Shared space similarity
    shared_sim = torch.cosine_similarity(
        mod_features[modalities[0]]['shared'],
        mod_features[modalities[1]]['shared'],
        dim=-1
    ).mean().item()
    
    # Private space similarity
    private_sim = torch.cosine_similarity(
        mod_features[modalities[0]]['private'],
        mod_features[modalities[1]]['private'],
        dim=-1
    ).mean().item()
    
    print(f"\nShared space 유사도: {shared_sim:.4f}")
    print(f"Private space 유사도: {private_sim:.4f}")
    print(f"\n판정:")
    print(f"  Shared:  {'✓ 높음 (좋음)' if shared_sim > 0.5 else '✗ 낮음 (나쁨)'}")
    print(f"  Private: {'✓ 낮음 (좋음)' if private_sim < 0.3 else '✗ 높음 (나쁨)'}")
    
    return shared_sim, private_sim


def verify_variance_distribution(mod_features):
    """
    방법 3: Variance 분포 확인
    Shared와 Private가 모두 정보를 담고 있는지 확인
    """
    print("\n" + "="*60)
    print("방법 3: Variance 분포 검증")
    print("="*60)
    
    for mod in mod_features:
        shared_var = mod_features[mod]['shared'].var(dim=0).mean().item()
        private_var = mod_features[mod]['private'].var(dim=0).mean().item()
        
        print(f"\n{mod} 모달리티:")
        print(f"  Shared variance:  {shared_var:.4f}")
        print(f"  Private variance: {private_var:.4f}")
        print(f"  비율: {shared_var/(shared_var+private_var):.2%} / {private_var/(shared_var+private_var):.2%}")
        
        if 0.3 < shared_var/(shared_var+private_var) < 0.7:
            print(f"  판정: ✓ 균형있게 분리됨")
        else:
            print(f"  판정: ✗ 한쪽에 치우침")


def visualize_feature_space(mod_features, save_path='focal_separation.png'):
    """
    방법 4: t-SNE 시각화
    Shared/Private space를 2D로 시각화
    """
    print("\n" + "="*60)
    print("방법 4: t-SNE 시각화")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    modalities = list(mod_features.keys())
    colors = ['red', 'blue']
    
    for i, space in enumerate(['shared', 'private']):
        ax = axes[i]
        
        # t-SNE
        all_features = []
        labels = []
        
        for j, mod in enumerate(modalities):
            features = mod_features[mod][space].cpu().numpy()
            if features.ndim == 3:
                features = features.reshape(-1, features.shape[-1])
            
            all_features.append(features)
            labels.extend([j] * len(features))
        
        all_features = np.concatenate(all_features, axis=0)
        
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(all_features)
        
        # Plot
        for j, mod in enumerate(modalities):
            mask = np.array(labels) == j
            ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                      c=colors[j], label=mod, alpha=0.6, s=10)
        
        ax.set_title(f'{space.capitalize()} Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 시각화 저장: {save_path}")
    print(f"\n판정 기준:")
    print(f"  Shared space: 모달리티들이 섞여 있어야 함 (공통 정보)")
    print(f"  Private space: 모달리티별로 분리되어 있어야 함 (고유 정보)")


def verify_reconstruction_quality(mod_features_original, mod_features_reconstructed):
    """
    방법 5: Reconstruction Quality
    Shared + Private를 합쳤을 때 원본과 유사한지 확인
    """
    print("\n" + "="*60)
    print("방법 5: Reconstruction Quality 검증")
    print("="*60)
    
    for mod in mod_features_original:
        # Concatenate shared + private
        reconstructed = torch.cat([
            mod_features_reconstructed[mod]['shared'],
            mod_features_reconstructed[mod]['private']
        ], dim=-1)
        
        # MSE
        mse = ((mod_features_original[mod] - reconstructed) ** 2).mean().item()
        
        # Cosine similarity
        cos_sim = torch.cosine_similarity(
            mod_features_original[mod].flatten(0, 1),
            reconstructed.flatten(0, 1),
            dim=-1
        ).mean().item()
        
        print(f"\n{mod} 모달리티:")
        print(f"  MSE: {mse:.4f}")
        print(f"  Cosine Similarity: {cos_sim:.4f}")
        print(f"  판정: {'✓ 정보 보존 잘 됨' if cos_sim > 0.95 else '✗ 정보 손실'}")


def main_verification(model, dataloader, device='cuda'):
    """
    통합 검증 실행
    """
    print("\n" + "🔍 " * 30)
    print("FOCAL Shared/Private Vector 분리 검증")
    print("🔍 " * 30)
    
    model.eval()
    
    # 데이터 로드
    batch = next(iter(dataloader))
    
    # Forward pass
    with torch.no_grad():
        # 두 개의 augmentation 결과
        mod_features1, mod_features2 = model(
            batch['aug1'].to(device),
            batch['aug2'].to(device),
            proj_head=False
        )
    
    # Split features
    from src.models.FOCALModules import split_features
    split_mod_features1 = split_features(mod_features1)
    split_mod_features2 = split_features(mod_features2)
    
    # 검증 실행
    print("\n📊 Aug1 Features 검증:")
    
    # 1. Orthogonality
    for mod in split_mod_features1:
        verify_orthogonality(
            split_mod_features1[mod]['shared'],
            split_mod_features1[mod]['private']
        )
    
    # 2. Cross-modal similarity
    verify_cross_modal_similarity(split_mod_features1)
    
    # 3. Variance distribution
    verify_variance_distribution(split_mod_features1)
    
    # 4. Visualization
    visualize_feature_space(split_mod_features1)
    
    # 5. Reconstruction quality
    verify_reconstruction_quality(mod_features1, split_mod_features1)
    
    print("\n" + "="*60)
    print("✅ 검증 완료!")
    print("="*60)


if __name__ == '__main__':
    print("이 스크립트는 FOCAL 모델의 학습된 checkpoint에서 사용하세요.")
    print("\n사용 예시:")
    print("  from verify_focal_separation import main_verification")
    print("  main_verification(model, dataloader, device='cuda')")

