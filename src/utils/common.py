"""공통 유틸리티 함수"""
import random
import numpy as np
import torch
import yaml
import os
from pathlib import Path


def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """YAML 설정 파일 저장"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_logger(log_dir, use_tensorboard=True):
    """로거 생성"""
    os.makedirs(log_dir, exist_ok=True)
    
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
        return writer
    return None


def save_results(results, save_path):
    """결과를 CSV로 저장"""
    import pandas as pd
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

