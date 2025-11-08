"""통합 학습 및 평가 스크립트"""
import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import set_seed, load_config, save_results, create_logger
from src.datamod import create_dataloaders
from src.metrics import compute_all_metrics
from src.baselines import CCALinearBaseline, CrossAutoEncoder, CrossAETrainer


def run_cca_linear(config, domain, seed):
    """CCA+Linear 베이스라인 실행"""
    print(f"\n{'='*50}")
    print(f"Running CCA+Linear Baseline - Domain: {domain}, Seed: {seed}")
    print(f"{'='*50}\n")
    
    set_seed(seed)
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        domain=domain,
        batch_size=config['train']['batch_size'],
        window_size=config['data']['window_T'],
        stride=config['data']['stride'],
        split_ratio=tuple(config['data']['split_ratio'].values()),
        normalize=config['data']['normalize'],
        root_dir='.'
    )
    
    # 모델 생성
    model = CCALinearBaseline(
        n_components=config['model']['cca_components'],
        alpha=config['train']['weight_decay']
    )
    
    # 학습
    print("Training CCA+Linear model...")
    model.fit(train_loader)
    
    # 평가
    print("Evaluating on test set...")
    results = model.evaluate(test_loader, compute_all_metrics)
    
    # 결과 출력
    print("\nResults:")
    print(f"A->B: MAE={results['A2B']['mae']:.4f}, DTW={results['A2B']['dtw']:.4f}, Pearson={results['A2B']['pearson']:.4f}")
    print(f"B->A: MAE={results['B2A']['mae']:.4f}, DTW={results['B2A']['dtw']:.4f}, Pearson={results['B2A']['pearson']:.4f}")
    
    # 결과 저장
    result_dict = {
        'model': 'CCA+Linear',
        'domain': domain,
        'seed': seed,
        'A2B_mae': results['A2B']['mae'],
        'A2B_dtw': results['A2B']['dtw'],
        'A2B_pearson': results['A2B']['pearson'],
        'B2A_mae': results['B2A']['mae'],
        'B2A_dtw': results['B2A']['dtw'],
        'B2A_pearson': results['B2A']['pearson'],
    }
    
    return result_dict


def run_cross_ae(config, domain, seed):
    """Cross-AutoEncoder 베이스라인 실행"""
    print(f"\n{'='*50}")
    print(f"Running Cross-AutoEncoder - Domain: {domain}, Seed: {seed}")
    print(f"{'='*50}\n")
    
    set_seed(seed)
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        domain=domain,
        batch_size=config['train']['batch_size'],
        window_size=config['data']['window_T'],
        stride=config['data']['stride'],
        split_ratio=tuple(config['data']['split_ratio'].values()),
        normalize=config['data']['normalize'],
        root_dir='.'
    )
    
    # 데이터 차원 확인
    sample_batch = next(iter(train_loader))
    dim_A = sample_batch['xA'].shape[-1]
    dim_B = sample_batch['xB'].shape[-1]
    
    print(f"Data dimensions: A={dim_A}, B={dim_B}")
    
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CrossAutoEncoder(
        dim_A=dim_A,
        dim_B=dim_B,
        hidden_dim=config['model']['ae_hidden_dim'],
        latent_dim=config['model']['ae_latent_dim']
    )
    
    trainer = CrossAETrainer(
        model=model,
        device=device,
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    # 학습
    print("Training Cross-AutoEncoder...")
    trainer.fit(
        train_loader,
        val_loader,
        epochs=config['train']['epochs'],
        patience=config['train']['early_stopping_patience']
    )
    
    # 평가
    print("Evaluating on test set...")
    results = trainer.evaluate(test_loader, compute_all_metrics)
    
    # 결과 출력
    print("\nResults:")
    print(f"A->B: MAE={results['A2B']['mae']:.4f}, DTW={results['A2B']['dtw']:.4f}, Pearson={results['A2B']['pearson']:.4f}")
    print(f"B->A: MAE={results['B2A']['mae']:.4f}, DTW={results['B2A']['dtw']:.4f}, Pearson={results['B2A']['pearson']:.4f}")
    
    # 결과 저장
    result_dict = {
        'model': 'Cross-AE',
        'domain': domain,
        'seed': seed,
        'A2B_mae': results['A2B']['mae'],
        'A2B_dtw': results['A2B']['dtw'],
        'A2B_pearson': results['A2B']['pearson'],
        'B2A_mae': results['B2A']['mae'],
        'B2A_dtw': results['B2A']['dtw'],
        'B2A_pearson': results['B2A']['pearson'],
    }
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate baselines')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='all',
                       choices=['cca', 'cross_ae', 'all'],
                       help='Which model to run')
    parser.add_argument('--domain', type=str, default='Agriculture',
                       help='Domain to use (Agriculture, Climate, etc.)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Seeds to use (overrides config)')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # Seeds
    seeds = args.seeds if args.seeds is not None else config['seeds']
    
    # 결과 저장 디렉토리
    os.makedirs('reports', exist_ok=True)
    
    # 실행
    all_results = []
    
    for seed in seeds:
        if args.model in ['cca', 'all']:
            try:
                result = run_cca_linear(config, args.domain, seed)
                all_results.append(result)
            except Exception as e:
                print(f"Error running CCA+Linear with seed {seed}: {e}")
        
        if args.model in ['cross_ae', 'all']:
            try:
                result = run_cross_ae(config, args.domain, seed)
                all_results.append(result)
            except Exception as e:
                print(f"Error running Cross-AE with seed {seed}: {e}")
    
    # 전체 결과 저장
    if len(all_results) > 0:
        save_path = f'reports/results_{args.domain}_{args.model}.csv'
        save_results(all_results, save_path)
        print(f"\n{'='*50}")
        print(f"All results saved to {save_path}")
        print(f"{'='*50}")
    else:
        print("No results to save!")


if __name__ == '__main__':
    main()

