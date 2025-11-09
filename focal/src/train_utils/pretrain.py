import os
import sys
import torch
import logging
import numpy as np

from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler
from train_utils.knn import compute_knn
from train_utils.model_selection import init_pretrain_framework
from train_utils.loss_calc_utils import calc_pretrain_loss
from general_utils.weight_utils import freeze_patch_embedding

# utils
from general_utils.time_utils import time_sync

# 시각화를 위한 import (focal 상위 디렉토리의 demo_visualization.py 사용)
focal_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, focal_root)
try:
    from demo_visualization import FOCALVisualizerDemo
    VISUALIZER_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ demo_visualization.py를 찾을 수 없습니다. 시각화는 비활성화됩니다.")
    VISUALIZER_AVAILABLE = False


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
    """
    The supervised training function for tbe backbone network,
    used in train of supervised mode or fine-tune of foundation models.
    """
    # Initialize contrastive model
    default_model = init_pretrain_framework(args, backbone_model)

    # Define the optimizer and learning rate scheduler
    optimizer = define_optimizer(args, default_model.parameters())
    lr_scheduler = define_lr_scheduler(args, optimizer)

    default_model = freeze_patch_embedding(args, default_model)

    # 시각화 초기화
    visualizer = None
    vis_folder = None
    if VISUALIZER_AVAILABLE:
        try:
            visualizer = FOCALVisualizerDemo()
            vis_folder = os.path.join(args.weight_folder, "visualizations")
            os.makedirs(vis_folder, exist_ok=True)
            logging.info(f"✅ 시각화 활성화됨. 저장 경로: {vis_folder}")
        except Exception as e:
            logging.warning(f"⚠️ 시각화 초기화 실패: {e}")
            visualizer = None
    
    # Training loop
    logging.info("---------------------------Start Pretraining Classifier-------------------------------")
    start = time_sync()
    best_val_loss = np.inf
    best_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_best.pt")
    latest_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_{args.stage}_latest.pt")

    for epoch in range(args.dataset_config[args.learn_framework]["pretrain_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        default_model.train()

        # training loop
        train_loss_list = []

        # regularization configuration
        for i, (time_loc_inputs, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # clear the gradients
            optimizer.zero_grad()

            # move to target device, FFT, and augmentations
            loss = calc_pretrain_loss(args, default_model, augmenter, loss_func, time_loc_inputs)

            # back propagation
            loss.backward()

            # update
            optimizer.step()
            train_loss_list.append(loss.item())

        if epoch % 10 == 0:
            # Use KNN classifier for validation
            knn_estimator = compute_knn(args, default_model.backbone, augmenter, train_dataloader)

            # validation and logging
            train_loss = np.mean(train_loss_list)
            val_acc, val_loss = val_and_logging(
                args,
                epoch,
                default_model,
                augmenter,
                val_dataloader,
                test_dataloader,
                loss_func,
                train_loss,
                estimator=knn_estimator,
            )

            # 🎨 시각화 추가 (Shared/Private 분리 확인)
            if visualizer is not None:
                try:
                    # validation batch 가져오기
                    val_batch, _ = next(iter(val_dataloader))
                    
                    # augmentation 적용
                    aug1 = augmenter.forward("fixed", val_batch)
                    aug2 = augmenter.forward("fixed", val_batch)
                    
                    # features 추출 (projection head 제외)
                    default_model.eval()
                    with torch.no_grad():
                        mod_features, _ = default_model(aug1, aug2, proj_head=False)
                    default_model.train()
                    
                    # 시각화 저장
                    vis_path = os.path.join(vis_folder, f"epoch_{epoch:04d}.png")
                    visualizer.visualize_all(mod_features, vis_path)
                    logging.info(f"📊 시각화 저장됨: {vis_path}")
                    
                except Exception as e:
                    logging.warning(f"⚠️ 시각화 생성 실패 (epoch {epoch}): {e}")

            # Save the latest model, only the backbone parameters are saved
            torch.save(default_model.backbone.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(default_model.backbone.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
