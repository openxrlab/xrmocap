# yapf: disable
from mmcv.utils import Registry
from torch.utils.data.dataloader import DataLoader

from xrmocap.data.dataset.builder import build_dataset

# yapf: enable

DATALOADERS = Registry('dataloader')
DATALOADERS.register_module(name='DataLoader', module=DataLoader)


def build_dataloader(cfg) -> DataLoader:
    """Build dataloader."""
    dataset = cfg.get('dataset', None)
    if isinstance(dataset, dict):
        dataset = build_dataset(dataset)
        cfg['dataset'] = dataset
    return DATALOADERS.build(cfg)
