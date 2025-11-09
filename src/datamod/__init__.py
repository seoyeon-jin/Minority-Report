from .dataset import TimeMMDDataset, create_dataloaders

try:
    from .dataset_v2 import TimeMMDDatasetV2
    __all__ = ['TimeMMDDataset', 'TimeMMDDatasetV2', 'create_dataloaders', 'custom_collate_fn']
except ImportError:
    __all__ = ['TimeMMDDataset', 'create_dataloaders']


def custom_collate_fn(batch):
    """
    메타데이터(dates, texts)를 적절히 처리하는 collate function
    
    Usage:
        from src.datamod import custom_collate_fn
        loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
    """
    import torch
    
    result = {}
    
    # Tensor 데이터는 기본 방식으로 stack
    result['xA'] = torch.stack([item['xA'] for item in batch])
    result['xB'] = torch.stack([item['xB'] for item in batch])
    result['maskA'] = torch.stack([item['maskA'] for item in batch])
    result['maskB'] = torch.stack([item['maskB'] for item in batch])
    
    # 메타데이터는 리스트로 유지
    if 'dates' in batch[0]:
        result['dates'] = [item['dates'] for item in batch]
        result['texts'] = [item['texts'] for item in batch]
        result['xA_raw'] = torch.stack([item['xA_raw'] for item in batch])
    
    return result

