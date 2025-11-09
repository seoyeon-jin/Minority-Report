"""여러 도메인 테스트 스크립트"""
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from src.datamod import TimeMMDDatasetV2, custom_collate_fn
from torch.utils.data import DataLoader


def test_domain(domain_name):
    """특정 도메인 테스트"""
    print(f"\n{'='*80}")
    print(f"🔍 Testing: {domain_name}")
    print(f"{'='*80}")
    
    try:
        # 데이터셋 생성
        dataset = TimeMMDDatasetV2(
            domain=domain_name,
            window_size=10,
            stride=5,
            split='train',
            text_mode='simple',
            return_metadata=True
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} windows")
        
        # 첫 샘플 확인
        sample = dataset[0]
        print(f"✓ Sample shape:")
        print(f"  - xA (Numerical): {sample['xA'].shape}")
        print(f"  - xB (Textual): {sample['xB'].shape}")
        print(f"  - Date range: {sample['dates'][0]} ~ {sample['dates'][-1]}")
        
        # DataLoader 테스트
        loader = DataLoader(dataset, batch_size=2, shuffle=False, 
                          collate_fn=custom_collate_fn)
        batch = next(iter(loader))
        
        print(f"✓ Batch shape:")
        print(f"  - xA: {batch['xA'].shape}")
        print(f"  - xB: {batch['xB'].shape}")
        
        # 샘플 데이터 확인
        print(f"✓ Sample data (first timestep):")
        print(f"  - Date: {batch['dates'][0][0]}")
        print(f"  - Numerical: {batch['xA'][0][0][:5]}...")  # 처음 5개만
        print(f"  - Text features: {batch['xB'][0][0]}")
        print(f"  - Masks: A={batch['maskA'][0][0].item()}, B={batch['maskB'][0][0].item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """여러 도메인 테스트"""
    print("\n" + "🌍 " * 30)
    print("Multi-Domain Test")
    print("🌍 " * 30)
    
    # 테스트할 도메인들
    domains = [
        'Agriculture',
        'Climate', 
        'Economy',
        'Energy',
        'Environment',
    ]
    
    results = {}
    
    for domain in domains:
        success = test_domain(domain)
        results[domain] = success
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 Test Results Summary")
    print("="*80)
    
    for domain, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{domain:20s}: {status}")
    
    all_passed = all(results.values())
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("\n" + "="*80)
    if all_passed:
        print(f"🎉 모든 도메인 테스트 통과! ({passed_count}/{total_count})")
    else:
        print(f"⚠️  일부 도메인 실패: {passed_count}/{total_count} 통과")
    print("="*80)
    
    print("\n💡 이제 모든 도메인에서 동일한 방식으로 사용할 수 있습니다:")
    print("  dataset = TimeMMDDatasetV2(domain='Agriculture', ...)")
    print("  dataset = TimeMMDDatasetV2(domain='Climate', ...)")
    print("  dataset = TimeMMDDatasetV2(domain='Economy', ...)")
    print("  # 날짜 컬럼이 자동으로 감지됩니다!")


if __name__ == '__main__':
    main()

