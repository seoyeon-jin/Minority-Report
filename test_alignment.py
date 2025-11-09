"""데이터 정렬 확인 스크립트"""
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from src.datamod import TimeMMDDatasetV2, custom_collate_fn


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
    
    # DataLoader로 배치 확인 (Custom collate function 사용)
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
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
    if len(batch['dates'][0]) > 2:
        print(f"  날짜: {batch['dates'][0][2]}")
        text_sample = batch['texts'][0][2]
        if len(text_sample) > 50:
            print(f"  텍스트: {text_sample[:50]}...")
        else:
            print(f"  텍스트: {text_sample}")
        print(f"  Numerical 값: {batch['xA'][0][2][:3]}")
        print(f"  Text 특성: {batch['xB'][0][2][:3]}")
    else:
        print("  (윈도우가 너무 작아 예시를 생략합니다)")
    
    print("\n" + "=" * 80)
    print("📝 중요: DataLoader 사용 시 custom_collate_fn 필요!")
    print("=" * 80)
    print("사용법:")
    print("  from src.datamod import TimeMMDDatasetV2, custom_collate_fn")
    print("  from torch.utils.data import DataLoader")
    print("  ")
    print("  dataset = TimeMMDDatasetV2(..., return_metadata=True)")
    print("  loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)")
    print("  ")
    print("  for batch in loader:")
    print("      dates = batch['dates']  # List[List[str]] - 배치 크기 x 윈도우 크기")
    print("      texts = batch['texts']  # List[List[str]]")
    print("      # dates[i][t]와 texts[i][t]는 xA[i,t], xB[i,t]와 완벽히 매칭됨!")


if __name__ == '__main__':
    test_alignment()

