import numpy as np
import scipy
from typing import List, Tuple

from xrmocap.utils.mvpose_utils import distribution


def get_conns(n_kps):
    conns = {}
    # for 13 keypoints configuration
    if n_kps == 13:
        conns['tree'] = [{} for i in range(13)]
        conns['tree'][0]['name'] = 'Nose'
        conns['tree'][0]['children'] = [1, 2, 7, 8]
        conns['tree'][1]['name'] = 'LSho'
        conns['tree'][1]['children'] = [3]
        conns['tree'][2]['name'] = 'RSho'
        conns['tree'][2]['children'] = [4]
        conns['tree'][3]['name'] = 'LElb'
        conns['tree'][3]['children'] = [5]
        conns['tree'][4]['name'] = 'RElb'
        conns['tree'][4]['children'] = [6]
        conns['tree'][5]['name'] = 'LWri'
        conns['tree'][5]['children'] = []
        conns['tree'][6]['name'] = 'RWri'
        conns['tree'][6]['children'] = []
        conns['tree'][7]['name'] = 'LHip'
        conns['tree'][7]['children'] = [9]
        conns['tree'][8]['name'] = 'RHip'
        conns['tree'][8]['children'] = [10]
        conns['tree'][9]['name'] = 'LKne'
        conns['tree'][9]['children'] = [11]
        conns['tree'][10]['name'] = 'RKne'
        conns['tree'][10]['children'] = [12]
        conns['tree'][11]['name'] = 'LAnk'
        conns['tree'][11]['children'] = []
        conns['tree'][12]['name'] = 'RAnk'
        conns['tree'][12]['children'] = []
    else:
        raise NotImplementedError('Other n_kps2d is not yet implemented.')
    return conns


def get_struct(conns: dict) -> List[dict]:
    """Get the pictorial structure.

    Args:
        conns (dict): The connection tree

    Returns:
        List[dict]: The selected keypoints connection information.
    """
    graph = conns['tree']
    level = np.zeros(len(graph))
    for i in range(len(graph)):
        queue = np.array(graph[i]['children'], dtype=np.int32)
        for j in range(queue.shape[0]):
            graph[queue[j]]['parent'] = i
        while queue.shape[0] != 0:
            level[queue[0]] = level[queue[0]] + 1
            queue = np.append(queue, graph[queue[0]]['children'])
            queue = np.delete(queue, 0)
            queue = np.array(queue, dtype=np.int32)
    trans_order = np.argsort(-level)
    limb = [{} for i in range(len(trans_order) - 1)]
    for i in range(len(trans_order) - 1):
        limb[i]['child'] = trans_order[i]
        limb[i]['parent'] = graph[limb[i]['child']]['parent']
        conn_id = distribution['kps2conns'][(limb[i]['child'],
                                             limb[i]['parent'])]
        limb[i]['bone_mean'] = np.array(distribution['mean'])[conn_id]
        limb[i]['bone_std'] = (np.array(distribution['std']) * 16)[conn_id]
    return limb


def get_prior(kps_idx: int, kps_cand_idx: int, parent_idx: int,
              parent_cand_idx: int, limb: List[dict],
              candidates: np.ndarray) -> np.float64:
    """Calculate the probability kps candidates and kps's parent candidates.

    Args:
        kps_idx (int): The keypoints index.
        kps_cand_idx (int): The kps_idx's candidates index.
        parent_idx (int): The kps_idx's parent index.
        parent_cand_idx (int): The parent's candidates index.
        limb (List[dict]): The selected keypoints connection information.
        candidates (np.ndarray): The candidates.

    Returns:
        np.float64: The probability.
    """
    limb_2_kps = [[], 8, 9, 4, 5, 0, 1, 10, 11, 6, 7, 2, 3]
    bone_std = limb[limb_2_kps[kps_idx]]['bone_std']
    bone_mean = limb[limb_2_kps[kps_idx]]['bone_mean']
    distance = np.linalg.norm(candidates[kps_idx][kps_cand_idx] -
                              candidates[parent_idx][parent_cand_idx])
    relative_error = np.abs(distance - bone_mean) / bone_std
    prior = scipy.stats.norm.sf(relative_error) * 2
    return prior


def get_max(kps_idx: int, parent_idx: int, parent_cand_idx: int,
            unary: np.ndarray, limb: List[dict],
            candidates: np.ndarray) -> Tuple[np.float64, np.int64]:
    """Get the maximum probability of candidate for each kps.

    Args:
        kps_idx (int): The keypoints index.
        parent_idx (int): The kps_idx's parent index.
        parent_cand_idx (int): The parent's candidates index.
        unary (np.ndarray)
        limb (List[dict]): The selected keypoints connection information.
        candidates (np.ndarray): The candidates.

    Returns:
        Tuple[np.float64, np.int64]:
        this_max: the maximum probability
        index: The candidate index of the maximum probability.
    """
    unary_sum = np.zeros(len(unary[kps_idx]))
    for kps_cand_idx in range(len(unary[kps_idx])):
        prior = get_prior(kps_idx, kps_cand_idx, parent_idx, parent_cand_idx,
                          limb, candidates)
        unary_sum[kps_cand_idx] = prior + unary[kps_idx][kps_cand_idx]
    this_max = np.max(unary_sum)
    index = np.where(unary_sum == np.max(unary_sum))[0][0]
    return this_max, index


def get_parent(kps_idx):
    child = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    parent_idx = [[], 0, 0, 1, 2, 3, 4, 0, 0, 7, 8, 9, 10]

    return parent_idx[child[kps_idx]]


def infer_kps3d_max_product(unary: np.ndarray, limb: List[dict],
                            candidates: np.ndarray) -> np.ndarray:
    """Inference the pictorial structure.

    Args:
        unary (np.ndarray): the probability of candidate for each kps.
        limb (List[dict]): The selected kps connection information.
        candidates (np.ndarray): The candidates.

    Returns:
        np.ndarray: The index of selected candidate for each kps.
    """
    n_kps = unary.shape[0]
    for kps_idx in range(n_kps - 1, 0, -1):
        parent_idx = get_parent(kps_idx)
        for parent_cand_idx in range(unary[parent_idx].shape[0]):
            m = get_max(kps_idx, parent_idx, parent_cand_idx, unary, limb,
                        candidates)
            unary[parent_idx][
                parent_cand_idx] = unary[parent_idx][parent_cand_idx] + m[0]
    # get the max index
    values = unary[0]
    selected_cand_idx = np.zeros(unary.shape[0], dtype=np.int64)
    selected_cand_idx[0] = values.argmax()
    for kps_idx in range(1, n_kps):
        parent_idx = get_parent(kps_idx)
        xn = get_max(kps_idx, parent_idx, selected_cand_idx[parent_idx], unary,
                     limb, candidates)
        selected_cand_idx[kps_idx] = xn[1]
    return selected_cand_idx
