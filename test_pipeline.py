"""파이프라인 테스트 스크립트"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import set_seed, load_config
from src.datamod import create_dataloaders
from src.metrics import compute_all_metrics
import numpy as np


def test_data_loading():
    """데이터 로딩 테스트"""
    print("\n" + "="*50)
    print("테스트 1: 데이터 로딩")
    print("="*50)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            domain='Agriculture',
            batch_size=8,
            window_size=64,
            stride=16,
            split_ratio=(0.6, 0.2, 0.2),
            normalize=True,
            root_dir='.'
        )
        
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
        print(f"✓ Test loader: {len(test_loader)} batches")
        
        # 샘플 배치 확인
        batch = next(iter(train_loader))
        print(f"\n배치 샘플:")
        print(f"  xA shape: {batch['xA'].shape}")
        print(f"  xB shape: {batch['xB'].shape}")
        print(f"  maskA shape: {batch['maskA'].shape}")
        print(f"  maskB shape: {batch['maskB'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False


def test_metrics():
    """메트릭 계산 테스트"""
    print("\n" + "="*50)
    print("테스트 2: 메트릭 계산")
    print("="*50)
    
    try:
        # 더미 데이터
        pred = np.random.randn(4, 32, 5)
        target = np.random.randn(4, 32, 5)
        mask = np.random.randint(0, 2, (4, 32))
        
        metrics = compute_all_metrics(pred, target, mask)
        
        print(f"✓ MAE: {metrics['mae']:.4f}")
        print(f"✓ DTW: {metrics['dtw']:.4f}")
        print(f"✓ Pearson: {metrics['pearson']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 오류: {e}")
        return False


def test_end_to_end():
    """End-to-end 파이프라인 테스트"""
    print("\n" + "="*50)
    print("테스트 3: End-to-End 파이프라인")
    print("="*50)
    
    try:
        set_seed(42)
        
        # 데이터 로더
        train_loader, val_loader, test_loader = create_dataloaders(
            domain='Agriculture',
            batch_size=8,
            window_size=64,
            stride=16,
            split_ratio=(0.6, 0.2, 0.2),
            normalize=True,
            root_dir='.'
        )
        
        print("✓ 데이터 로더 생성 완료")
        
        # Train/Val/Test 순회
        for split_name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
            batch_count = 0
            for batch in loader:
                batch_count += 1
                if batch_count >= 2:  # 처음 2개 배치만
                    break
            print(f"✓ {split_name} 배치 순회 OK ({batch_count} batches)")
        
        # 메트릭 계산
        test_batch = next(iter(test_loader))
        metrics = compute_all_metrics(
            test_batch['xA'],
            test_batch['xB'],
            test_batch['maskA']
        )
        print(f"✓ 메트릭 계산 OK (MAE={metrics['mae']:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """전체 테스트 실행"""
    print("\n" + "#"*50)
    print("# Time-MMD 파이프라인 테스트")
    print("#"*50)
    
    results = []
    
    # 테스트 실행
    results.append(("데이터 로딩", test_data_loading()))
    results.append(("메트릭 계산", test_metrics()))
    results.append(("End-to-End", test_end_to_end()))
    
    # 결과 요약
    print("\n" + "="*50)
    print("테스트 결과 요약")
    print("="*50)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ 모든 테스트 통과!")
        print("="*50)
        print("\n다음 명령어로 실제 학습을 시작할 수 있습니다:")
        print("  python src/train.py --domain Agriculture --model cca")
        print("  python src/train.py --domain Agriculture --model cross_ae")
        print("  python src/train.py --domain Agriculture --model all")
    else:
        print("✗ 일부 테스트 실패")
        print("="*50)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())

