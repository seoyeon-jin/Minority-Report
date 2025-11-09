"""
FOCAL 시각화 데모 - 바로 실행 가능!
더미 데이터로 시각화 예시를 생성합니다.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def create_dummy_features(batch_size=64, seq_len=4, feature_dim=256):
    """
    더미 멀티모달 features 생성
    실제 FOCAL 학습 결과를 시뮬레이션
    """
    print("🎲 더미 데이터 생성 중...")
    
    # 두 개 모달리티 (예: seismic, audio)
    mod_features = {}
    
    for i, mod_name in enumerate(['seismic', 'audio']):
        # 전체 feature 생성
        features = torch.randn(batch_size, seq_len, feature_dim)
        
        # Shared part에 모달리티 간 상관관계 추가
        shared_dim = feature_dim // 2
        common_signal = torch.randn(batch_size, seq_len, shared_dim)
        
        # Shared part는 공통 + 노이즈
        features[:, :, :shared_dim] = common_signal + torch.randn_like(common_signal) * 0.3
        
        # Private part는 모달리티별로 다르게
        features[:, :, shared_dim:] = torch.randn(batch_size, seq_len, shared_dim) + i * 2
        
        mod_features[mod_name] = features
    
    print(f"✓ 생성 완료: {len(mod_features)} 모달리티")
    for mod, feat in mod_features.items():
        print(f"  - {mod}: {feat.shape}")
    
    return mod_features


def split_features(mod_features):
    """Feature를 shared/private로 분리"""
    split_mod_features = {}
    
    for mod in mod_features:
        b, seq, dim = mod_features[mod].shape
        split_dim = dim // 2
        split_mod_features[mod] = {
            "shared": mod_features[mod][:, :, 0:split_dim],
            "private": mod_features[mod][:, :, split_dim:],
        }
    
    return split_mod_features


class FOCALVisualizerDemo:
    """FOCAL 시각화 (데모용)"""
    
    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.colors = {
            'shared': '#FF6B6B',
            'private': '#4ECDC4',
            'modA': '#FFE66D',
            'modB': '#95E1D3'
        }
    
    def visualize_all(self, mod_features, save_path='focal_demo.png'):
        """전체 시각화"""
        split_features_dict = split_features(mod_features)
        
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        print("\n🎨 시각화 생성 중...")
        
        # 1. t-SNE
        print("  - t-SNE 계산 중...")
        self._plot_tsne_comparison(split_features_dict, fig, gs[0:2, 0:2])
        
        # 2. Similarity Heatmap
        print("  - Similarity Heatmap...")
        self._plot_similarity_heatmap(split_features_dict, fig, gs[0, 2])
        
        # 3. Orthogonality
        print("  - Orthogonality 분포...")
        self._plot_orthogonality_distribution(split_features_dict, fig, gs[1, 2])
        
        # 4. Variance
        print("  - Variance 비율...")
        self._plot_variance_ratio(split_features_dict, fig, gs[2, 0])
        
        # 5. Correlation
        print("  - Feature Correlation...")
        self._plot_feature_correlation(split_features_dict, fig, gs[2, 1])
        
        # 6. Summary
        print("  - 종합 점수 계산...")
        self._plot_summary_score(split_features_dict, fig, gs[2, 2])
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 시각화 저장 완료: {save_path}")
        plt.close()
        
        return save_path
    
    def _plot_tsne_comparison(self, split_features, fig, gs):
        """t-SNE 시각화"""
        ax_shared = fig.add_subplot(gs[0, 0])
        ax_private = fig.add_subplot(gs[0, 1])
        ax_combined = fig.add_subplot(gs[1, :])
        
        modalities = list(split_features.keys())
        
        for ax, space, title in [(ax_shared, 'shared', 'Shared Space'),
                                  (ax_private, 'private', 'Private Space')]:
            all_features = []
            labels = []
            
            for i, mod in enumerate(modalities):
                features = split_features[mod][space].cpu().numpy()
                features = features.reshape(-1, features.shape[-1])
                
                all_features.append(features)
                labels.extend([mod] * len(features))
            
            all_features = np.concatenate(all_features, axis=0)
            
            # t-SNE
            perplexity = min(30, len(all_features) // 3)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedded = tsne.fit_transform(all_features)
            
            # Plot
            for i, mod in enumerate(modalities):
                mask = np.array([l == mod for l in labels])
                color = self.colors['modA'] if i == 0 else self.colors['modB']
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                          c=color, label=mod, alpha=0.6, s=30, 
                          edgecolors='white', linewidth=0.5)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            
            judgment = ("✓ 모달리티가 섞여야 함\n(공통 정보)" if space == 'shared' 
                       else "✓ 모달리티가 분리되어야 함\n(고유 정보)")
            ax.text(0.02, 0.98, judgment, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Combined PCA
        for space, marker in [('shared', 'o'), ('private', '^')]:
            for mod_idx, mod in enumerate(modalities):
                features = split_features[mod][space].cpu().numpy()
                features = features.reshape(-1, features.shape[-1])
                
                pca = PCA(n_components=2)
                embedded = pca.fit_transform(features)
                
                color = self.colors['shared'] if space == 'shared' else self.colors['private']
                ax_combined.scatter(embedded[:, 0], embedded[:, 1],
                                  c=color, marker=marker,
                                  label=f'{mod}-{space}', alpha=0.5, s=30)
        
        ax_combined.set_title('Combined View (PCA)', fontsize=14, fontweight='bold')
        ax_combined.legend(loc='best', ncol=2, frameon=True, shadow=True)
        ax_combined.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_similarity_heatmap(self, split_features, fig, gs):
        """Similarity Heatmap"""
        ax = fig.add_subplot(gs)
        modalities = list(split_features.keys())
        
        matrix = np.zeros((4, 4))
        labels = []
        
        for i, mod1 in enumerate(modalities):
            for space1 in ['shared', 'private']:
                idx1 = i * 2 + (0 if space1 == 'shared' else 1)
                labels.append(f'{mod1}\n{space1}')
                
                for j, mod2 in enumerate(modalities):
                    for space2 in ['shared', 'private']:
                        idx2 = j * 2 + (0 if space2 == 'shared' else 1)
                        
                        feat1 = split_features[mod1][space1].flatten(0, -2)
                        feat2 = split_features[mod2][space2].flatten(0, -2)
                        
                        sim = torch.cosine_similarity(feat1, feat2, dim=-1).mean().item()
                        matrix[idx1, idx2] = sim
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=labels, yticklabels=labels,
                   vmin=-0.5, vmax=1.0, center=0.3,
                   cbar_kws={'label': 'Cosine Similarity'},
                   ax=ax, square=True)
        
        ax.set_title('Cross-Similarity Matrix', fontsize=12, fontweight='bold')
    
    def _plot_orthogonality_distribution(self, split_features, fig, gs):
        """Orthogonality 분포"""
        ax = fig.add_subplot(gs)
        
        all_sims = []
        for mod in split_features:
            shared = split_features[mod]['shared']
            private = split_features[mod]['private']
            
            shared_norm = shared / (torch.norm(shared, dim=-1, keepdim=True) + 1e-8)
            private_norm = private / (torch.norm(private, dim=-1, keepdim=True) + 1e-8)
            
            cos_sim = (shared_norm * private_norm).sum(dim=-1)
            all_sims.extend(cos_sim.abs().cpu().numpy().flatten())
        
        ax.hist(all_sims, bins=50, color=self.colors['shared'], 
               alpha=0.7, edgecolor='black', linewidth=0.5)
        
        mean_sim = np.mean(all_sims)
        ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_sim:.3f}')
        ax.axvline(0.1, color='green', linestyle=':', linewidth=2,
                  label='목표: <0.1', alpha=0.7)
        
        ax.set_xlabel('|Cosine Similarity|', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Orthogonality (Shared ⊥ Private)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        judgment = "✓ 잘 분리됨" if mean_sim < 0.1 else "✗ 분리 부족"
        color = 'green' if mean_sim < 0.1 else 'red'
        ax.text(0.95, 0.95, judgment, transform=ax.transAxes,
               fontsize=11, ha='right', va='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    def _plot_variance_ratio(self, split_features, fig, gs):
        """Variance 비율"""
        ax = fig.add_subplot(gs)
        
        modalities = list(split_features.keys())
        shared_vars = []
        private_vars = []
        
        for mod in modalities:
            shared_var = split_features[mod]['shared'].var(dim=0).mean().item()
            private_var = split_features[mod]['private'].var(dim=0).mean().item()
            
            total = shared_var + private_var
            shared_vars.append(shared_var / total * 100)
            private_vars.append(private_var / total * 100)
        
        x = np.arange(len(modalities))
        width = 0.7
        
        ax.bar(x, shared_vars, width, label='Shared',
              color=self.colors['shared'], alpha=0.8, edgecolor='black')
        ax.bar(x, private_vars, width, bottom=shared_vars,
              label='Private', color=self.colors['private'], 
              alpha=0.8, edgecolor='black')
        
        for i, (s, p) in enumerate(zip(shared_vars, private_vars)):
            ax.text(i, s/2, f'{s:.1f}%', ha='center', va='center', fontweight='bold')
            ax.text(i, s + p/2, f'{p:.1f}%', ha='center', va='center', fontweight='bold')
        
        ax.set_ylabel('Variance (%)', fontsize=11)
        ax.set_title('Variance Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modalities)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhspan(30, 70, alpha=0.1, color='green')
    
    def _plot_feature_correlation(self, split_features, fig, gs):
        """Feature Correlation"""
        ax = fig.add_subplot(gs)
        
        mod = list(split_features.keys())[0]
        
        shared = split_features[mod]['shared'].flatten(0, -2).cpu().numpy()
        private = split_features[mod]['private'].flatten(0, -2).cpu().numpy()
        
        shared_mean = shared.mean(axis=0)
        private_mean = private.mean(axis=0)
        
        ax.scatter(shared_mean, private_mean, alpha=0.6, s=50,
                  c=self.colors['shared'], edgecolors='black', linewidth=0.5)
        
        lims = [min(shared_mean.min(), private_mean.min()),
                max(shared_mean.max(), private_mean.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect Correlation')
        
        ax.set_xlabel('Shared Feature Magnitude', fontsize=11)
        ax.set_ylabel('Private Feature Magnitude', fontsize=11)
        ax.set_title(f'Feature Correlation ({mod})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        corr = np.corrcoef(shared_mean, private_mean)[0, 1]
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_summary_score(self, split_features, fig, gs):
        """종합 점수"""
        ax = fig.add_subplot(gs)
        ax.axis('off')
        
        modalities = list(split_features.keys())
        scores = {}
        
        # 1. Orthogonality
        ortho_sims = []
        for mod in modalities:
            shared = split_features[mod]['shared']
            private = split_features[mod]['private']
            shared_norm = shared / (torch.norm(shared, dim=-1, keepdim=True) + 1e-8)
            private_norm = private / (torch.norm(private, dim=-1, keepdim=True) + 1e-8)
            cos_sim = (shared_norm * private_norm).sum(dim=-1)
            ortho_sims.append(cos_sim.abs().mean().item())
        
        ortho_score = max(0, (0.1 - np.mean(ortho_sims)) / 0.1 * 100)
        scores['Orthogonality'] = ortho_score
        
        # 2. Shared similarity
        shared_sim = torch.cosine_similarity(
            split_features[modalities[0]]['shared'].flatten(0, -2),
            split_features[modalities[1]]['shared'].flatten(0, -2),
            dim=-1
        ).mean().item()
        shared_score = max(0, (shared_sim - 0.3) / 0.5 * 100)
        scores['Shared Similarity'] = shared_score
        
        # 3. Private dissimilarity
        private_sim = torch.cosine_similarity(
            split_features[modalities[0]]['private'].flatten(0, -2),
            split_features[modalities[1]]['private'].flatten(0, -2),
            dim=-1
        ).mean().item()
        private_score = max(0, (0.5 - private_sim) / 0.5 * 100)
        scores['Private Dissimilarity'] = private_score
        
        # 4. Variance balance
        var_scores = []
        for mod in modalities:
            shared_var = split_features[mod]['shared'].var(dim=0).mean().item()
            private_var = split_features[mod]['private'].var(dim=0).mean().item()
            ratio = shared_var / (shared_var + private_var)
            var_scores.append(max(0, 100 - abs(ratio - 0.5) * 200))
        variance_score = np.mean(var_scores)
        scores['Variance Balance'] = variance_score
        
        overall = np.mean(list(scores.values()))
        
        # 시각화
        y_pos = 0.9
        ax.text(0.5, y_pos, '📊 종합 평가', ha='center', va='top',
               fontsize=14, fontweight='bold')
        
        y_pos -= 0.15
        for name, score in scores.items():
            color = 'green' if score > 70 else ('orange' if score > 40 else 'red')
            bar_width = score / 100 * 0.6
            
            rect = plt.Rectangle((0.2, y_pos-0.05), bar_width, 0.04,
                                facecolor=color, alpha=0.6, edgecolor='black')
            ax.add_patch(rect)
            
            ax.text(0.15, y_pos, name + ':', ha='right', va='center', fontsize=10)
            ax.text(0.85, y_pos, f'{score:.1f}', ha='left', va='center',
                   fontsize=10, fontweight='bold')
            
            y_pos -= 0.12
        
        y_pos -= 0.05
        overall_color = 'green' if overall > 70 else ('orange' if overall > 40 else 'red')
        ax.text(0.5, y_pos, f'Overall: {overall:.1f}/100', ha='center', va='center',
               fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=overall_color, alpha=0.3))
        
        y_pos -= 0.15
        if overall > 70:
            judgment = "✅ 우수: 잘 분리됨"
        elif overall > 40:
            judgment = "⚠️ 보통: 개선 필요"
        else:
            judgment = "❌ 불량: 분리 실패"
        
        ax.text(0.5, y_pos, judgment, ha='center', va='center',
               fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("🎨 FOCAL Shared/Private 분리 시각화 데모")
    print("="*60)
    
    # 1. 더미 데이터 생성
    mod_features = create_dummy_features(
        batch_size=64,
        seq_len=4,
        feature_dim=256
    )
    
    # 2. 시각화
    visualizer = FOCALVisualizerDemo(figsize=(20, 12))
    save_path = visualizer.visualize_all(mod_features, 'focal_demo_result.png')
    
    print("\n" + "="*60)
    print("✅ 완료!")
    print("="*60)
    print(f"\n📁 결과 파일: {save_path}")
    print("\n💡 실제 사용 시:")
    print("   1. 학습된 FOCAL 모델에서 mod_features 추출")
    print("   2. visualizer.visualize_all(mod_features, 'result.png')")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

