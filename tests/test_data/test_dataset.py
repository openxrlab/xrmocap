import mmcv

from xrmocap.data.dataloader.builder import build_dataloader
from xrmocap.data.dataset.builder import build_dataset


def test_build_shelf_dataset():
    dataset_config = dict(
        mmcv.Config.fromfile('configs/modules/data/dataset/' +
                             'shelf_unittest.py'))
    dataset = build_dataset(dataset_config)
    assert len(dataset) > 0


def test_load_shelf_dataset_w_perception2d():
    dataset_config = dict(
        mmcv.Config.fromfile('configs/modules/data/dataset/' +
                             'shelf_unittest.py'))
    dataloader_config = dict(
        type='DataLoader', dataset=dataset_config, batch_size=1, num_workers=1)
    dataloader = build_dataloader(dataloader_config)
    dataloader.dataset[0]
    iter_count = 0
    for batch_idx, batch_data in enumerate(dataloader):
        # mview img shape: batch_size, n_v, h, w, c
        assert len(batch_data[0].shape) == 5
        # K shape: batch_size, n_v, 3, 3
        assert len(batch_data[1].shape) == 4
        assert batch_data[1].shape[-2:] == (3, 3)
        # R shape: batch_size, n_v, 3, 3
        assert len(batch_data[2].shape) == 4
        assert batch_data[2].shape[-2:] == (3, 3)
        # T shape: batch_size, n_v, 3
        assert len(batch_data[3].shape) == 3
        assert batch_data[3].shape[-1] == 3
        # kps3d shape: batch_size, n_person, n_kps, 4
        assert len(batch_data[4].shape) == 4
        assert batch_data[4].shape[-1] == 4
        # end_of_clip shape: batch_size
        assert len(batch_data[5].shape) == 1
        assert batch_data[5][0].item() == \
            (batch_idx == len(dataloader.dataset)-1)
        # kw_data
        kw_data = batch_data[6]
        # bbox shape: batch_size, n_v, n_f, p, 5
        assert 'bbox2d' in kw_data
        assert 'kps2d' in kw_data
        iter_count = batch_idx + 1
    expect_n_batch = len(dataloader.dataset)
    assert iter_count == expect_n_batch


def test_load_shelf_dataset_wo_perception2d():
    dataset_config = dict(
        mmcv.Config.fromfile('configs/modules/data/dataset/' +
                             'shelf_unittest.py'))
    dataset_config['bbox_convention'] = None
    dataset_config['kps2d_convention'] = None
    dataloader_config = dict(
        type='DataLoader', dataset=dataset_config, batch_size=2, num_workers=2)
    dataloader = build_dataloader(dataloader_config)
    dataloader.dataset[0]
    iter_count = 0
    for batch_idx, batch_data in enumerate(dataloader):
        # mview img shape: batch_size, n_v, h, w, c
        assert len(batch_data[0].shape) == 5
        # K shape: batch_size, n_v, 3, 3
        assert len(batch_data[1].shape) == 4
        assert batch_data[1].shape[-2:] == (3, 3)
        # R shape: batch_size, n_v, 3, 3
        assert len(batch_data[2].shape) == 4
        assert batch_data[2].shape[-2:] == (3, 3)
        # T shape: batch_size, n_v, 3
        assert len(batch_data[3].shape) == 3
        assert batch_data[3].shape[-1] == 3
        # kps3d shape: batch_size, n_person, n_kps, 4
        assert len(batch_data[4].shape) == 4
        assert batch_data[4].shape[-1] == 4
        # end_of_clip shape: batch_size
        assert len(batch_data[5].shape) == 1
        iter_count = batch_idx + 1
    expect_n_batch = int(
        len(dataloader.dataset) / 2) + len(dataloader.dataset) % 2
    assert iter_count == expect_n_batch
