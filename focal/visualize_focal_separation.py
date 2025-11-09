"""
FOCAL Shared/Specific Vector 분리 시각화 스크립트
시각화 중심으로 직관적으로 분리 품질 확인
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class FOCALSeparationVisualizer:
    """FOCAL의 Shared/Private 분리를 시각화"""
    
    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.colors = {
            'shared': '#FF6B6B',
            'private': '#4ECDC4',
            'modA': '#FFE66D',
            'modB': '#95E1D3'
        }
    
    def visualize_all(self, mod_features, save_path='focal_analysis.png'):
        """전체 시각화 실행"""
        from src.models.FOCALModules import split_features
        split_features_dict = split_features(mod_features)
        
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. t-SNE 시각화 (좌상단 2x2)
        self._plot_tsne_comparison(split_features_dict, fig, gs[0:2, 0:2])
        
        # 2. Similarity Heatmap (우상단)
        self._plot_similarity_heatmap(split_features_dict, fig, gs[0, 2])
        
        # 3. Orthogonality 분포 (우중단)
        self._plot_orthogonality_distribution(split_features_dict, fig, gs[1, 2])
        
        # 4. Variance 비율 (좌하단)
        self._plot_variance_ratio(split_features_dict, fig, gs[2, 0])
        
        # 5. Feature Correlation (중하단)
        self._plot_feature_correlation(split_features_dict, fig, gs[2, 1])
        
        # 6. Summary Score (우하단)
        self._plot_summary_score(split_features_dict, fig, gs[2, 2])
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 전체 분석 저장: {save_path}")
        plt.close()
        
        return save_path
    
    def _plot_tsne_comparison(self, split_features, fig, gs):
        """1. t-SNE로 Shared vs Private 비교"""
        ax_shared = fig.add_subplot(gs[0, 0])
        ax_private = fig.add_subplot(gs[0, 1])
        ax_combined = fig.add_subplot(gs[1, :])
        
        modalities = list(split_features.keys())
        
        for ax, space, title in [(ax_shared, 'shared', 'Shared Space'),
                                  (ax_private, 'private', 'Private Space')]:
            # 데이터 준비
            all_features = []
            labels = []
            colors = []
            
            for i, mod in enumerate(modalities):
                features = split_features[mod][space].cpu().numpy()
                if features.ndim == 3:
                    features = features.reshape(-1, features.shape[-1])
                
                all_features.append(features)
                labels.extend([f'{mod}'] * len(features))
                colors.extend([self.colors['modA'] if i == 0 else self.colors['modB']] * len(features))
            
            all_features = np.concatenate(all_features, axis=0)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
            embedded = tsne.fit_transform(all_features)
            
            # Plot
            for i, mod in enumerate(modalities):
                mask = np.array([l == mod for l in labels])
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                          c=self.colors['modA'] if i == 0 else self.colors['modB'],
                          label=mod, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            
            # 판정 텍스트
            if space == 'shared':
                judgment = "✓ 모달리티가 섞여야 함\n(공통 정보)"
            else:
                judgment = "✓ 모달리티가 분리되어야 함\n(고유 정보)"
            ax.text(0.02, 0.98, judgment, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Combined view
        for space_idx, (space, marker) in enumerate([('shared', 'o'), ('private', '^')]):
            for mod_idx, mod in enumerate(modalities):
                features = split_features[mod][space].cpu().numpy()
                if features.ndim == 3:
                    features = features.reshape(-1, features.shape[-1])
                
                # PCA for combined view
                pca = PCA(n_components=2)
                embedded = pca.fit_transform(features)
                
                color = self.colors['shared'] if space == 'shared' else self.colors['private']
                ax_combined.scatter(embedded[:, 0], embedded[:, 1],
                                  c=color, marker=marker,
                                  label=f'{mod}-{space}', alpha=0.5, s=30)
        
        ax_combined.set_title('Combined View (PCA)', fontsize=14, fontweight='bold')
        ax_combined.legend(loc='best', ncol=2, frameon=True, shadow=True)
        ax_combined.grid(True, alpha=0.3, linestyle='--')
        ax_combined.set_xlabel('PC 1')
        ax_combined.set_ylabel('PC 2')
    
    def _plot_similarity_heatmap(self, split_features, fig, gs):
        """2. Similarity Heatmap"""
        ax = fig.add_subplot(gs)
        
        modalities = list(split_features.keys())
        
        # Similarity matrix 계산
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
        
        # Heatmap
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=labels, yticklabels=labels,
                   vmin=-0.5, vmax=1.0, center=0.3,
                   cbar_kws={'label': 'Cosine Similarity'},
                   ax=ax, square=True)
        
        ax.set_title('Cross-Similarity Matrix', fontsize=12, fontweight='bold')
        
        # 판정 기준 추가
        ax.text(1.5, -0.5, 
               "✓ Shared-Shared: 높아야 함 (>0.5)\n✓ Private-Private: 낮아야 함 (<0.3)",
               fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def _plot_orthogonality_distribution(self, split_features, fig, gs):
        """3. Orthogonality 분포"""
        ax = fig.add_subplot(gs)
        
        all_sims = []
        modalities = list(split_features.keys())
        
        for mod in modalities:
            shared = split_features[mod]['shared']
            private = split_features[mod]['private']
            
            # Normalize
            shared_norm = shared / (torch.norm(shared, dim=-1, keepdim=True) + 1e-8)
            private_norm = private / (torch.norm(private, dim=-1, keepdim=True) + 1e-8)
            
            # Cosine similarity
            cos_sim = (shared_norm * private_norm).sum(dim=-1)
            all_sims.extend(cos_sim.abs().cpu().numpy().flatten())
        
        # Histogram
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
        
        # 판정
        judgment = "✓ 잘 분리됨" if mean_sim < 0.1 else "✗ 분리 부족"
        color = 'green' if mean_sim < 0.1 else 'red'
        ax.text(0.95, 0.95, judgment, transform=ax.transAxes,
               fontsize=11, ha='right', va='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    def _plot_variance_ratio(self, split_features, fig, gs):
        """4. Variance 비율"""
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
        width = 0.35
        
        bars1 = ax.bar(x, shared_vars, width, label='Shared',
                      color=self.colors['shared'], alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, private_vars, width, bottom=shared_vars,
                      label='Private', color=self.colors['private'], 
                      alpha=0.8, edgecolor='black')
        
        # 값 표시
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
        
        # 목표 범위
        ax.axhspan(30, 70, alpha=0.1, color='green', label='목표 범위')
    
    def _plot_feature_correlation(self, split_features, fig, gs):
        """5. Feature 간 상관관계"""
        ax = fig.add_subplot(gs)
        
        modalities = list(split_features.keys())
        mod = modalities[0]  # 첫 번째 모달리티만
        
        shared = split_features[mod]['shared'].flatten(0, -2).cpu().numpy()
        private = split_features[mod]['private'].flatten(0, -2).cpu().numpy()
        
        # Correlation scatter
        # 각 feature dimension의 평균값
        shared_mean = shared.mean(axis=0)
        private_mean = private.mean(axis=0)
        
        ax.scatter(shared_mean, private_mean, alpha=0.6, s=50,
                  c=self.colors['shared'], edgecolors='black', linewidth=0.5)
        
        # Diagonal line (완전 상관)
        lims = [min(shared_mean.min(), private_mean.min()),
                max(shared_mean.max(), private_mean.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect Correlation')
        
        ax.set_xlabel('Shared Feature Magnitude', fontsize=11)
        ax.set_ylabel('Private Feature Magnitude', fontsize=11)
        ax.set_title(f'Feature Correlation ({mod})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 상관계수
        corr = np.corrcoef(shared_mean, private_mean)[0, 1]
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_summary_score(self, split_features, fig, gs):
        """6. 종합 점수"""
        ax = fig.add_subplot(gs)
        ax.axis('off')
        
        modalities = list(split_features.keys())
        
        # 점수 계산
        scores = {}
        
        # 1. Orthogonality score
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
        
        # 2. Shared similarity score
        shared_sim = torch.cosine_similarity(
            split_features[modalities[0]]['shared'].flatten(0, -2),
            split_features[modalities[1]]['shared'].flatten(0, -2),
            dim=-1
        ).mean().item()
        shared_score = max(0, (shared_sim - 0.3) / 0.5 * 100)
        scores['Shared Similarity'] = shared_score
        
        # 3. Private dissimilarity score
        private_sim = torch.cosine_similarity(
            split_features[modalities[0]]['private'].flatten(0, -2),
            split_features[modalities[1]]['private'].flatten(0, -2),
            dim=-1
        ).mean().item()
        private_score = max(0, (0.5 - private_sim) / 0.5 * 100)
        scores['Private Dissimilarity'] = private_score
        
        # 4. Variance balance score
        var_scores = []
        for mod in modalities:
            shared_var = split_features[mod]['shared'].var(dim=0).mean().item()
            private_var = split_features[mod]['private'].var(dim=0).mean().item()
            ratio = shared_var / (shared_var + private_var)
            var_scores.append(max(0, 100 - abs(ratio - 0.5) * 200))
        variance_score = np.mean(var_scores)
        scores['Variance Balance'] = variance_score
        
        # Overall score
        overall = np.mean(list(scores.values()))
        
        # 시각화
        y_pos = 0.9
        ax.text(0.5, y_pos, '📊 종합 평가', ha='center', va='top',
               fontsize=14, fontweight='bold')
        
        y_pos -= 0.15
        for name, score in scores.items():
            color = 'green' if score > 70 else ('orange' if score > 40 else 'red')
            bar_width = score / 100 * 0.6
            
            # 막대
            rect = plt.Rectangle((0.2, y_pos-0.05), bar_width, 0.04,
                                facecolor=color, alpha=0.6, edgecolor='black')
            ax.add_patch(rect)
            
            # 텍스트
            ax.text(0.15, y_pos, name + ':', ha='right', va='center', fontsize=10)
            ax.text(0.85, y_pos, f'{score:.1f}', ha='left', va='center',
                   fontsize=10, fontweight='bold')
            
            y_pos -= 0.12
        
        # Overall
        y_pos -= 0.05
        overall_color = 'green' if overall > 70 else ('orange' if overall > 40 else 'red')
        ax.text(0.5, y_pos, f'Overall: {overall:.1f}/100', ha='center', va='center',
               fontsize=13, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=overall_color, alpha=0.3))
        
        # 판정
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


def visualize_from_checkpoint(checkpoint_path, dataloader, save_dir='focal_visualizations'):
    """체크포인트에서 모델을 로드하고 시각화"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n🎨 FOCAL 분리 시각화 시작...")
    
    # 모델 로드 (실제 로드 코드는 프로젝트에 맞게 수정)
    # model = load_model(checkpoint_path)
    # model.eval()
    
    # 데이터 로드
    # batch = next(iter(dataloader))
    # with torch.no_grad():
    #     mod_features1, _ = model(batch['aug1'], batch['aug2'])
    
    # 시각화 (예시 데이터)
    print("⚠️  실제 사용 시 위 주석을 해제하고 모델과 데이터를 로드하세요.")
    print("\n예시 사용법:")
    print("  visualizer = FOCALSeparationVisualizer()")
    print("  visualizer.visualize_all(mod_features, save_path='result.png')")


if __name__ == '__main__':
    print("="*60)
    print("FOCAL Shared/Private 분리 시각화 도구")
    print("="*60)
    print("\n📌 사용 방법:")
    print("\n1. 학습 중 사용:")
    print("   from visualize_focal_separation import FOCALSeparationVisualizer")
    print("   visualizer = FOCALSeparationVisualizer()")
    print("   visualizer.visualize_all(mod_features, 'epoch_10.png')")
    print("\n2. 체크포인트에서 사용:")
    print("   visualize_from_checkpoint('best.pth', dataloader)")
    print("\n" + "="*60)

