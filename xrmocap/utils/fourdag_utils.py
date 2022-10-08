import json
import math
import numpy as np


class LimbInfo():

    def __init__(self, kps_convention) -> None:
        self.kps_convention = kps_convention
        with open('./weight/limb_info.json', 'r') as f:
            self.info_dict = json.load(f)[self.kps_convention]

    def get_kps_number(self):
        """get keypoints number."""
        return self.info_dict['n_kps']

    def get_paf_number(self):
        """get paf number."""
        return self.info_dict['n_pafs']

    def get_shape_size(self):
        """get the prior shape number."""
        return self.info_dict['shape_size']

    def get_kps_parent(self):
        """get keypoints parent list."""
        return self.info_dict['kps_parent']

    def get_shape_blend(self):
        """get shape blend."""
        return self.info_dict['shape_blend']

    def get_kps_prior(self):
        """get prior keypoints."""
        return self.info_dict['m_kps']

    def get_hierarchy_map(self):
        """get hierarchy map for keypoints."""
        return self.info_dict['hierarchy_map']

    def get_paf_dict(self):
        """get paf dict."""
        return self.info_dict['paf_dict']


def welsch(c, x):
    x = x / c
    return 1 - math.exp(-x * x / 2)


def line2linedist(pa, raya, pb, rayb):
    if abs(np.vdot(raya, rayb)) < 1e-5:
        return point2linedist(pa, pb, raya)
    else:
        ve = np.cross(raya, rayb)
        ve = ve / np.linalg.norm(ve)
        ve = abs(np.vdot((pa - pb), ve))
        return ve


def point2linedist(pa, pb, ray):
    ve = np.cross(pa - pb, ray)
    return np.linalg.norm(ve)


def skew(vec):
    m_skew = np.zeros((3, 3), dtype=np.float32)
    m_skew = np.array(
        [0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0],
        dtype=np.float32).reshape((3, 3))
    return m_skew


def rodrigues(vec):
    theta = np.linalg.norm(vec)
    identity = np.identity(3, dtype=np.float32)
    if abs(theta) < 1e-5:
        return identity
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        r = vec / theta
        return c * identity + np.matmul((1 - c) * r.reshape(
            (-1, 1)), r.reshape((1, -1))) + s * skew(r)


def rodrigues_jacobi(vec):
    theta = np.linalg.norm(vec)
    d_skew = np.zeros((3, 9), dtype=np.float32)
    d_skew[0, 5] = d_skew[1, 6] = d_skew[2, 1] = -1
    d_skew[0, 7] = d_skew[1, 2] = d_skew[2, 3] = 1
    if abs(theta) < 1e-5:
        return -d_skew
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        c1 = 1 - c
        itheta = 1 / theta
        r = vec / theta
        rrt = np.matmul(r.reshape((-1, 1)), r.reshape((1, -1)))
        m_skew = skew(r)
        identity = np.identity(3, dtype=np.float32)
        drrt = np.array([
            r[0] + r[0], r[1], r[2], r[1], 0, 0, r[2], 0, 0, 0, r[0], 0, r[0],
            r[1] + r[1], r[2], 0, r[2], 0, 0, 0, r[0], 0, 0, r[1], r[0], r[1],
            r[2] + r[2]
        ],
                        dtype=np.float32).reshape((3, 9))
        jaocbi = np.zeros((3, 9), dtype=np.float32)
        a = np.zeros((5, 1), dtype=np.float32)
        for i in range(3):
            a = np.array([
                -s * r[i], (s - 2 * c1 * itheta) * r[i], c1 * itheta,
                (c - s * itheta) * r[i], s * itheta
            ],
                         dtype=np.float32).reshape((5, 1))
            for j in range(3):
                for k in range(3):

                    jaocbi[i, k + k + k + j] = (
                        a[0] * identity[j, k] + a[1] * rrt[j, k] +
                        a[2] * drrt[i, j + j + j + k] + a[3] * m_skew[j, k] +
                        a[4] * d_skew[i, j + j + j + k])
        return jaocbi
