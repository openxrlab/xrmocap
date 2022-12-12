import copy
import numpy
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pathlib import Path
from typing import Union

from xrmocap.utils.distribute_utils import is_main_process


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def match_name_keywords(n, name_keywords):
    """Match the keys in two dictionaries."""
    for b in name_keywords:
        if b in n:
            return True
    return False


def get_model_name(model, resnet_layer):
    name = f'{model}_{resnet_layer}'
    return name


def get_directory(state: str = 'train',
                  output_dir: str = 'output',
                  cfg_name: str = 'mvp_train',
                  dataset: str = 'dataset',
                  model: str = 'multi_view_pose_transformer',
                  resnet_layer: int = 50):
    """This function generates the log file directory and output file
    directory.

    Args:
        state (str, optional): Train or test mode. Defaults to 'train'.
        output_dir (str, optional): Output directory. Defaults to 'output'.
        cfg_name (str, optional): Name of the configuration file. Defaults
            to 'mvp_train'.
        dataset (str, optional): Name of the dataset. Defaults to 'dataset'.
        model (str, optional): Name of the model. Defaults to
            'multi_view_pose_transformer'.
        resnet_layer (int, optional): Number of layers in the backbone Resnet.
            Defaults to 50.
    """
    this_dir = Path(os.path.dirname(__file__))
    root_output_dir = (this_dir / '..' / '..' / output_dir).resolve()

    if not root_output_dir.exists() and is_main_process():
        root_output_dir.mkdir()

    model = get_model_name(model, resnet_layer)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    # get final output dir
    final_output_dir = os.path.join(root_output_dir, dataset, model, cfg_name)

    if (not os.path.exists(final_output_dir)):
        os.makedirs(final_output_dir)

    # get logger file under final output dir
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{cfg_name}_{time_str}_{state}.log'
    log_file = os.path.join(final_output_dir, log_file)

    return log_file, final_output_dir


def save_checkpoint(states,
                    is_best,
                    output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


@torch.no_grad()
def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ]), norm_type)
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def convert_result_to_kps(pred):
    """Convert MVP predict result to Keypoints object."""
    pred = pred[0]
    pred = pred[pred[:, 0, 3] >= 0]
    per_frame_kps3d = pred[:, :, :4]
    per_frame_kps3d[:, :, -1] = pred[:, :, -1]
    n_person = len(per_frame_kps3d)
    return n_person, per_frame_kps3d


def absolute2norm(absolute_coords, grid_size, grid_center):
    """convert abosolute keypoint coordinates to normalized(0-1)
    coordinates."""
    device = absolute_coords.device
    grid_size = grid_size.to(device=device)
    grid_center = grid_center.to(device=device)
    norm_coords = (absolute_coords - grid_center + grid_size / 2.0) / grid_size
    return norm_coords


def norm2absolute(norm_coords, grid_size, grid_center):
    """convert normalized (0-1) keypoint coordinates to abosolute
    coordinates."""
    device = norm_coords.device
    grid_size = grid_size.to(device=device)
    grid_center = grid_center.to(device=device)
    loc = norm_coords * grid_size + grid_center - grid_size / 2.0
    return loc


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def set_cudnn(benchmark: bool, deterministic: bool, enable: bool):
    cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.enabled = enable


def process_dict(src_dict,
                 device: Union[None, str] = 'cuda',
                 dummy_dim: Union[None, int] = None,
                 dtype: Union[None, str] = 'tensor'):
    for k, v in src_dict.items():
        if isinstance(v, numpy.ndarray) or isinstance(v, torch.Tensor):
            if dtype == 'tensor':
                v = torch.tensor(v)
            if dummy_dim is not None:
                v = v.unsqueeze(dummy_dim)
            if device is not None:
                v = v.to(device)
            src_dict[k] = v
        elif isinstance(v, dict):
            src_dict[k] = process_dict(v, device, dummy_dim, dtype)
    return src_dict
