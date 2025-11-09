"""
FOCAL 학습 코드에 시각화 통합하기
실제 학습된 hidden state를 시각화합니다.
"""

# ============================================================
# 📍 위치 1: focal/src/train_utils/pretrain.py
# ============================================================

PRETRAIN_INTEGRATION = """
# focal/src/train_utils/pretrain.py 파일의 상단에 추가

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from demo_visualization import FOCALVisualizerDemo

def pretrain(
    args,
    backbone_model,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    loss_func,
    num_batches,
):
    # 기존 코드...
    default_model = init_pretrain_framework(args, backbone_model)
    optimizer = define_optimizer(args, default_model.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)
    
    # ✨ 시각화 초기화 추가
    visualizer = FOCALVisualizerDemo(figsize=(20, 12))
    vis_dir = os.path.join(args.output_folder, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    logging.info(f"시각화 저장 경로: {vis_dir}")
    
    # Training loop
    for epoch in range(args.dataset_config[args.learn_framework]["pretrain_lr_scheduler"]["train_epochs"]):
        
        # ... 학습 코드 ...
        default_model.train()
        train_loss_list = []
        
        for i, (time_loc_inputs, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            loss = calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
        
        # ✨ 매 10 epoch마다 시각화 추가
        if epoch % 10 == 0:
            # KNN validation (기존 코드)
            knn_estimator = compute_knn(args, default_model.backbone, augmenter, train_dataloader)
            
            train_loss = np.mean(train_loss_list)
            val_acc, val_loss = val_and_logging(...)
            
            # ✨✨✨ 시각화 추가 ✨✨✨
            logging.info(f"Epoch {epoch}: 시각화 생성 중...")
            try:
                # Validation batch 가져오기
                val_iter = iter(val_dataloader)
                val_batch, _ = next(val_iter)
                
                # Augmentation
                aug_freq_loc_inputs_1 = augmenter.forward("fixed", val_batch)
                aug_freq_loc_inputs_2 = augmenter.forward("fixed", val_batch)
                
                # Forward pass로 features 추출 (proj_head=False!)
                default_model.eval()
                with torch.no_grad():
                    mod_features, _ = default_model(
                        aug_freq_loc_inputs_1,
                        aug_freq_loc_inputs_2,
                        proj_head=False  # ⚠️ 중요: projection head 전의 features
                    )
                
                # 시각화 저장
                save_path = os.path.join(vis_dir, f'epoch_{epoch:04d}.png')
                visualizer.visualize_all(mod_features, save_path)
                logging.info(f"   ✓ 시각화 저장: {save_path}")
                
            except Exception as e:
                logging.warning(f"   ✗ 시각화 실패: {e}")
            
            # 모델 다시 train mode로
            default_model.train()
            
            # 체크포인트 저장 (기존 코드)
            torch.save(default_model.backbone.state_dict(), latest_weight)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(default_model.backbone.state_dict(), best_weight)
        
        lr_scheduler.step(epoch)
    
    logging.info(f"✅ 학습 완료! 시각화 결과: {vis_dir}")
"""


# ============================================================
# 📍 위치 2: Loss 값 확인 (선택사항)
# ============================================================

LOSS_LOGGING = """
# focal/src/models/loss.py의 FOCALLoss.forward() 메서드 끝부분

def forward(self, mod_features1, mod_features2, index=None):
    # ... 기존 loss 계산 코드 ...
    
    # Step 2: shared space contrastive loss
    shared_contrastive_loss = 0
    # ... 계산 ...
    
    # Step 3: private space contrastive loss
    private_contrastive_loss = 0
    # ... 계산 ...
    
    # Step 4: temporal consistency loss
    temporal_consistency_loss = 0
    # ... 계산 ...
    
    # Step 5: orthogonality loss
    orthogonality_loss = 0
    # ... 계산 ...
    
    loss = (
        shared_contrastive_loss * self.config["shared_contrastive_loss_weight"]
        + private_contrastive_loss * self.config["private_contrastive_loss_weight"]
        + orthogonality_loss * self.config["orthogonal_loss_weight"]
        + temporal_consistency_loss * self.config["rank_loss_weight"]
    )
    
    # ✨ Loss 값들을 dict로 반환하도록 수정 (디버깅용)
    loss_dict = {
        'total': loss.item(),
        'shared': shared_contrastive_loss.item(),
        'private': private_contrastive_loss.item(),
        'orthogonal': orthogonality_loss.item(),
        'temporal': temporal_consistency_loss.item(),
    }
    
    # 기존: return loss
    # 수정: return loss, loss_dict  # dict도 함께 반환
    return loss  # 또는 loss와 dict 둘 다 반환
"""


# ============================================================
# 📍 실제 적용 방법
# ============================================================

def print_integration_guide():
    """통합 가이드 출력"""
    
    print("\n" + "="*80)
    print("🎨 FOCAL 시각화 통합 가이드")
    print("="*80)
    
    print("\n📍 Step 1: pretrain.py 수정")
    print("-" * 80)
    print("파일: focal/src/train_utils/pretrain.py")
    print("\n수정 위치:")
    print("  1) 파일 상단에 import 추가")
    print("  2) pretrain() 함수 내부에 visualizer 초기화")
    print("  3) epoch 루프에 시각화 코드 추가 (epoch % 10 == 0 부분)")
    
    print("\n" + PRETRAIN_INTEGRATION)
    
    print("\n" + "="*80)
    print("📍 Step 2: 실행")
    print("-" * 80)
    print("""
# 학습 시작
python focal/src/train.py \\
    --dataset MOD \\
    --model DeepSense \\
    --learn_framework FOCAL \\
    --stage pretrain

# 학습 중 자동으로 생성됨:
# - visualizations/epoch_0000.png
# - visualizations/epoch_0010.png
# - visualizations/epoch_0020.png
# - ...
""")
    
    print("\n" + "="*80)
    print("📍 Step 3: 결과 확인")
    print("-" * 80)
    print("""
# 생성된 시각화 확인
ls -lh output/MOD_DeepSense_pretrain/visualizations/

# 특정 epoch 확인
open output/MOD_DeepSense_pretrain/visualizations/epoch_0100.png

# GIF로 변환 (선택사항)
cd output/MOD_DeepSense_pretrain/visualizations/
convert -delay 50 -loop 0 epoch_*.png training_progress.gif
""")
    
    print("\n" + "="*80)
    print("📍 핵심 포인트")
    print("-" * 80)
    print("""
✅ Features 추출:
   mod_features, _ = default_model(aug1, aug2, proj_head=False)
   
   ⚠️ proj_head=False가 중요! 
      True: projection head 거친 features (loss 계산용)
      False: 원본 backbone features (시각화용)

✅ mod_features 구조:
   {
       'seismic': tensor(batch, seq, dim),
       'audio': tensor(batch, seq, dim)
   }
   
   - 이게 바로 shared/private로 split되는 features!
   - split_features()로 반으로 나눔
   - 앞 절반: shared, 뒤 절반: private

✅ Loss 위치:
   focal/src/models/loss.py의 FOCALLoss.forward()
   
   - shared_contrastive_loss
   - private_contrastive_loss
   - orthogonality_loss
   - temporal_consistency_loss
""")
    
    print("\n" + "="*80)
    print("🔧 빠른 테스트")
    print("-" * 80)
    print("""
# 1. demo로 먼저 테스트
cd /Users/sheoyonjin/Desktop/Minority-Report/focal
python demo_visualization.py

# 2. 위 pretrain.py 수정 적용

# 3. 학습 시작
python src/train.py --dataset MOD --learn_framework FOCAL --stage pretrain

# 4. 학습 중 visualizations/ 폴더 확인
watch -n 10 ls -lh output/*/visualizations/
""")
    
    print("\n" + "="*80)


# ============================================================
# 📍 간단 버전 (복사-붙여넣기용)
# ============================================================

SIMPLE_VERSION = """
# ===============================================
# pretrain.py 에 이 코드만 추가하면 끝!
# ===============================================

# 1) 파일 상단
from demo_visualization import FOCALVisualizerDemo
visualizer = FOCALVisualizerDemo()
vis_dir = 'visualizations'
os.makedirs(vis_dir, exist_ok=True)

# 2) epoch 루프 내부 (epoch % 10 == 0 부분)
if epoch % 10 == 0:
    # 기존 validation 코드...
    
    # 시각화 추가 (5줄만!)
    val_batch, _ = next(iter(val_dataloader))
    aug1 = augmenter.forward("fixed", val_batch)
    aug2 = augmenter.forward("fixed", val_batch)
    with torch.no_grad():
        features, _ = default_model(aug1, aug2, proj_head=False)
    visualizer.visualize_all(features, f'{vis_dir}/epoch_{epoch:04d}.png')
"""


if __name__ == '__main__':
    print_integration_guide()
    
    print("\n" + "🚀" * 40)
    print("간단 버전 (복사-붙여넣기)")
    print("🚀" * 40)
    print(SIMPLE_VERSION)
    
    print("\n" + "="*80)
    print("✅ 준비 완료!")
    print("="*80)
    print("\n위 코드를 focal/src/train_utils/pretrain.py에 추가하세요.")
    print("학습 시작하면 자동으로 시각화가 생성됩니다! 🎨")
    print("\n" + "="*80)

