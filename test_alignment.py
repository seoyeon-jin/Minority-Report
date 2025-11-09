"""데이터 정렬 확인 스크립트"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.datamod.dataset_v2 import TimeMMDDatasetV2


def test_alignment():
    """데이터 정렬 확인"""
    print("\n" + "🔍 " * 20)
    print("Time-MMD 데이터 정렬 테스트")
    print("🔍 " * 20)
    
    # 데이터셋 생성
    dataset = TimeMMDDatasetV2(
        domain='Agriculture',
        window_size=10,  # 작은 윈도우로 테스트
        stride=5,
        split='train',
        text_mode='simple',
        return_metadata=True
    )
    
    print(f"\n✓ Dataset loaded: {len(dataset)} windows")
    
    # 첫 번째 샘플 확인
    dataset.verify_alignment(idx=0, n_steps=5)
    
    # DataLoader로 배치 확인
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    print("\n" + "=" * 80)
    print("📦 Batch Information")
    print("=" * 80)
    print(f"Batch size: {batch['xA'].shape[0]}")
    print(f"xA shape: {batch['xA'].shape}")
    print(f"xB shape: {batch['xB'].shape}")
    print(f"Dates (first sample, first 3 steps): {batch['dates'][0][:3]}")
    
    # 텍스트 출력 (길이 체크)
    first_text = batch['texts'][0][0]
    if len(first_text) > 100:
        print(f"Texts (first sample, first step): {first_text[:100]}...")
    else:
        print(f"Texts (first sample, first step): {first_text}")
    
    print("\n✅ 데이터 정렬 확인 완료!")
    print("\n💡 사용법:")
    print("  for batch in dataloader:")
    print("      xA = batch['xA']        # (B, T, dA) - numerical")
    print("      xB = batch['xB']        # (B, T, dB) - textual features")
    print("      dates = batch['dates']  # List[List[str]] - 날짜 (문자열)")
    print("      texts = batch['texts']  # List[List[str]] - 원본 텍스트")
    print("      # dates[i][t]와 texts[i][t]는 xA[i,t], xB[i,t]와 매칭됨!")
    print("\n🔍 실제 사용 예시:")
    print("  # 첫 번째 샘플의 세 번째 시점")
    print(f"  날짜: {batch['dates'][0][2]}")
    print(f"  텍스트: {batch['texts'][0][2][:50]}...")
    print(f"  Numerical 값: {batch['xA'][0][2][:3]}")
    print(f"  Text 특성: {batch['xB'][0][2][:3]}")


if __name__ == '__main__':
    test_alignment()

