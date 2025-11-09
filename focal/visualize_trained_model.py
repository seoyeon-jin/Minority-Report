"""
학습된 FOCAL 모델 시각화 스크립트
학습 완료 후 checkpoint에서 모델을 로드하고 시각화합니다.
"""
import torch
import sys
from pathlib import Path

# FOCAL 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

from demo_visualization import FOCALVisualizerDemo


def visualize_checkpoint(checkpoint_path, dataloader, device='cuda', save_path='trained_model_analysis.png'):
    """
    학습된 체크포인트 시각화
    
    Args:
        checkpoint_path: 체크포인트 파일 경로 (예: 'checkpoints/best_model.pth')
        dataloader: Validation 또는 Test DataLoader
        device: 'cuda' 또는 'cpu'
        save_path: 시각화 결과 저장 경로
    """
    print("\n" + "="*60)
    print("🎨 학습된 FOCAL 모델 시각화")
    print("="*60)
    
    # 1. 체크포인트 로드
    print(f"\n📂 체크포인트 로드 중: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 체크포인트 정보 출력
    if 'epoch' in checkpoint:
        print(f"   - Epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"   - Best Accuracy: {checkpoint['best_acc']:.4f}")
    
    # 2. 모델 로드 (실제 사용 시 주석 해제)
    print("\n⚠️  주의: 아래 모델 로드 코드를 실제 프로젝트에 맞게 수정하세요!")
    print("""
    # 예시 코드:
    from src.models.FOCALModules import FOCAL
    from src.models.DeepSense import DeepSense  # 또는 다른 백본
    
    # 백본 생성
    backbone = DeepSense(args)
    
    # FOCAL 모델 생성
    model = FOCAL(args, backbone)
    
    # State dict 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    """)
    
    # 데모용 더미 모델 (실제 사용 시 위 코드로 교체)
    print("\n⚠️  현재는 더미 데이터로 데모를 실행합니다.")
    print("    실제 사용 시 위 주석을 해제하고 모델을 로드하세요.\n")
    
    # 3. 데이터 로드
    print("📊 데이터 로드 중...")
    batch = next(iter(dataloader))
    
    # 4. Forward pass (실제 사용 시 주석 해제)
    print("🔮 Forward pass...")
    print("""
    # 예시 코드:
    with torch.no_grad():
        # 입력 준비
        aug_freq_input1 = batch['aug1'].to(device)
        aug_freq_input2 = batch['aug2'].to(device)
        
        # Forward pass
        mod_features1, mod_features2 = model(
            aug_freq_input1,
            aug_freq_input2,
            proj_head=False  # 중요: projection head 전의 features 필요
        )
        
        # 첫 번째 augmentation의 features 사용
        mod_features = mod_features1
    """)
    
    # 데모용 더미 features (실제 사용 시 위 코드로 교체)
    from demo_visualization import create_dummy_features
    print("⚠️  더미 features 생성 중...")
    mod_features = create_dummy_features(batch_size=64, seq_len=4, feature_dim=256)
    
    # 5. 시각화
    print("\n🎨 시각화 생성 중...")
    visualizer = FOCALVisualizerDemo(figsize=(20, 12))
    result_path = visualizer.visualize_all(mod_features, save_path)
    
    print("\n" + "="*60)
    print("✅ 완료!")
    print("="*60)
    print(f"\n📁 결과 저장: {result_path}")
    print("\n💡 다음 단계:")
    print("   1. 위 주석 처리된 코드를 실제 프로젝트에 맞게 수정")
    print("   2. 체크포인트와 데이터로더 경로 설정")
    print("   3. 스크립트 재실행")
    print("\n" + "="*60)
    
    return result_path


def visualize_during_training_example():
    """
    학습 루프에 통합하는 예시 코드
    """
    print("\n" + "="*60)
    print("📚 학습 중 시각화 통합 예시")
    print("="*60)
    
    code = '''
# focal/src/train_utils/pretrain.py 또는 finetune.py 수정

from demo_visualization import FOCALVisualizerDemo
import os

# 학습 시작 전에 초기화
visualizer = FOCALVisualizerDemo(figsize=(20, 12))
vis_dir = 'visualizations'
os.makedirs(vis_dir, exist_ok=True)

# 학습 루프
for epoch in range(args.dataset_config[args.learn_framework]["pretrain_lr_scheduler"]["train_epochs"]):
    
    # ... 학습 코드 ...
    model.train()
    for batch in train_loader:
        # 학습 진행
        loss = criterion(...)
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        # ... Validation 코드 ...
        
        # ✨ 매 10 epoch마다 시각화 (또는 원하는 주기)
        if epoch % 10 == 0:
            print(f"\\n🎨 Epoch {epoch}: 시각화 생성 중...")
            
            # Validation batch에서 features 추출
            val_batch = next(iter(val_loader))
            aug1 = val_batch['aug1'].to(device)
            aug2 = val_batch['aug2'].to(device)
            
            # Forward pass
            mod_features, _ = model(aug1, aug2, proj_head=False)
            
            # 시각화 저장
            save_path = os.path.join(vis_dir, f'epoch_{epoch:04d}.png')
            visualizer.visualize_all(mod_features, save_path)
            print(f"   ✓ 저장: {save_path}")
    
    # 체크포인트 저장
    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, f'checkpoints/epoch_{epoch:04d}.pth')

print("\\n✅ 학습 완료!")
print(f"시각화 결과: {vis_dir}/ 폴더 확인")
'''
    
    print(code)
    print("\n" + "="*60)


def quick_visualization_guide():
    """빠른 사용 가이드"""
    print("\n" + "🚀" * 30)
    print("빠른 시작 가이드")
    print("🚀" * 30)
    
    print("\n📍 Case 1: 학습 완료 후 체크포인트 분석")
    print("-" * 60)
    print("""
from visualize_trained_model import visualize_checkpoint

# 체크포인트와 데이터로더 준비
checkpoint_path = 'checkpoints/best_model.pth'
val_loader = create_your_val_dataloader()  # 프로젝트의 dataloader

# 시각화 실행
visualize_checkpoint(
    checkpoint_path=checkpoint_path,
    dataloader=val_loader,
    device='cuda',
    save_path='final_analysis.png'
)
""")
    
    print("\n📍 Case 2: 학습 중 주기적 시각화")
    print("-" * 60)
    print("""
from demo_visualization import FOCALVisualizerDemo

visualizer = FOCALVisualizerDemo()

# 학습 루프 내부
for epoch in range(epochs):
    # ... 학습 ...
    
    if epoch % 10 == 0:  # 매 10 epoch
        with torch.no_grad():
            mod_features, _ = model(val_batch1, val_batch2, proj_head=False)
            visualizer.visualize_all(mod_features, f'epoch_{epoch}.png')
""")
    
    print("\n📍 Case 3: 여러 체크포인트 비교")
    print("-" * 60)
    print("""
import glob

checkpoints = sorted(glob.glob('checkpoints/epoch_*.pth'))

for ckpt in checkpoints:
    epoch = int(ckpt.split('_')[-1].split('.')[0])
    
    # 각 체크포인트 시각화
    visualize_checkpoint(
        checkpoint_path=ckpt,
        dataloader=val_loader,
        save_path=f'comparison/epoch_{epoch:04d}.png'
    )

print("✓ 모든 체크포인트 시각화 완료!")
""")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    print("\n" + "🎯" * 30)
    print("학습된 FOCAL 모델 시각화 도구")
    print("🎯" * 30)
    
    # 사용 가이드 출력
    quick_visualization_guide()
    
    print("\n" + "="*60)
    print("💡 실제 사용 예시")
    print("="*60)
    print("""
# 1. 이 스크립트 수정:
#    - 모델 로드 코드 주석 해제
#    - 프로젝트에 맞게 수정

# 2. 실행:
from visualize_trained_model import visualize_checkpoint

checkpoint_path = 'checkpoints/best_model.pth'
val_loader = your_val_dataloader  # 실제 데이터로더

visualize_checkpoint(checkpoint_path, val_loader)

# 3. 결과 확인:
#    trained_model_analysis.png 파일 생성됨!
""")
    
    print("\n학습 중 통합 예시:")
    visualize_during_training_example()
    
    print("\n" + "="*60)
    print("✅ 준비 완료! 위 가이드를 참고하여 사용하세요.")
    print("="*60)

