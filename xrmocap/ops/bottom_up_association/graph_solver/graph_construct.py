# yapf: disable
import logging
import numpy as np
from typing import Union

from xrmocap.utils.fourdag_utils import LimbInfo, line2linedist, point2linedist

# yapf: enable


class Camera():

    def __init__(self, cam_param) -> None:
        super().__init__()
        c_K = cam_param.intrinsic33()
        c_T = np.array(cam_param.get_extrinsic_t())
        c_R = np.array(cam_param.get_extrinsic_r())
        c_Ki = np.linalg.inv(c_K)
        self.c_Rt_Ki = np.matmul(c_R.T, c_Ki)
        self.Pos = -np.matmul(c_R.T, c_T)

    def cal_ray(self, uv):
        var = -self.c_Rt_Ki.dot(np.append(uv, 1).T)
        return var / np.linalg.norm(var)


class GraphConstruct():

    def __init__(self,
                 kps_convention='fourdag_19',
                 n_views=5,
                 max_epi_dist: float = 0.15,
                 max_temp_dist: float = 0.2,
                 normalize_edges: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """

        Args:
            kps_convention (str):
                The name of destination convention.
            n_views (int):
                views number of dataset
            n_kps (int):
                keypoints number
            n_pafs (int):
                paf number
            max_epi_dist (float):
                maximal epipolar distance to be accepted
            max_temp_dist (float):
                maximal temporal tracking distance to be accepted
            normalize_edges (bool):
                indicator to normalize all edges
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = logger
        self.n_views = n_views
        self.limb_info = LimbInfo(kps_convention)
        self.n_kps = self.limb_info.get_kps_number()
        self.n_pafs = self.limb_info.get_paf_number()
        self.max_epi_dist = max_epi_dist
        self.max_temp_dist = max_temp_dist
        self.normalize_edges = normalize_edges
        self.paf_dict = self.limb_info.get_paf_dict()
        self.m_epi_edges = {
            i: {
                j: {k: -1
                    for k in range(self.n_views)}
                for j in range(self.n_views)
            }
            for i in range(self.n_kps)
        }
        self.m_temp_edges = {
            i: {j: -1
                for j in range(self.n_views)}
            for i in range(self.n_kps)
        }
        self.m_kps_rays = {
            i: {j: []
                for j in range(self.n_kps)}
            for i in range(self.n_views)
        }
        self.m_bone_nodes = {
            i: {j: []
                for j in range(self.n_views)}
            for i in range(self.n_pafs)
        }
        self.m_bone_epi_edges = {
            i: {
                j: {k: []
                    for k in range(self.n_views)}
                for j in range(self.n_views)
            }
            for i in range(self.n_pafs)
        }
        self.m_bone_temp_edges = {
            i: {j: []
                for j in range(self.n_views)}
            for i in range(self.n_pafs)
        }

        self.cameras = []
        self.last_multi_kps3d = dict()

    def set_cameras(self, cameras_param):
        for view in range(len(cameras_param)):
            self.cameras.append(Camera(cameras_param[view]))

    def __call__(self, kps2d, pafs, last_multi_kps3d=dict):
        """Match people id from different cameras.

        Args:
            kps2d (list): 2D keypoints
            pafs (list): part affine field
            last_multi_kps3d (dict): 3D keypoints of last frame

        Returns:
            graph (dict): the constructed 4D graph
        """
        self.kps2d = kps2d
        self.pafs = pafs
        self.last_multi_kps3d = last_multi_kps3d
        self.construct_graph()

        return dict(
            m_epi_edges=self.m_epi_edges,
            m_temp_edges=self.m_temp_edges,
            m_bone_nodes=self.m_bone_nodes,
            m_bone_epi_edges=self.m_bone_epi_edges,
            m_bone_temp_edges=self.m_bone_temp_edges)

    def construct_graph(self):
        self._calculate_kps_rays()
        self._calculate_paf_edges()
        self._calculate_epi_edges()
        self._calculate_temp_edges()

        self._calculate_bone_nodes()
        self._calculate_bone_epi_edges()
        self._calculate_bone_temp_edges()

    def _calculate_kps_rays(self):
        for view in range(self.n_views):
            cam = self.cameras[view]
            for kps_id in range(self.n_kps):
                self.m_kps_rays[view][kps_id] = []
                kps = self.kps2d[view][kps_id]
                for kps_candidate in range(len(kps)):
                    self.m_kps_rays[view][kps_id].append(
                        cam.cal_ray(kps[kps_candidate][:2]))

    def _calculate_paf_edges(self):
        if self.normalize_edges:
            for paf_id in range(self.n_pafs):
                for detection in self.pafs:
                    pafs = detection[paf_id]
                    if np.sum(pafs) > 0:
                        row_factor = np.clip(pafs.sum(1), 1.0, None)
                        col_factor = np.clip(pafs.sum(0), 1.0, None)
                        for i in range(len(row_factor)):
                            pafs[i] /= row_factor[i]
                        for j in range(len(col_factor)):
                            pafs[:, j] /= col_factor[j]
                    detection[paf_id] = pafs

    def _calculate_epi_edges(self):
        for kps_id in range(self.n_kps):
            for view1 in range(self.n_views - 1):
                cam1 = self.cameras[view1]
                for view2 in range(view1 + 1, self.n_views):
                    cam2 = self.cameras[view2]
                    kps1 = self.kps2d[view1][kps_id]
                    kps2 = self.kps2d[view2][kps_id]
                    ray1 = self.m_kps_rays[view1][kps_id]
                    ray2 = self.m_kps_rays[view2][kps_id]

                    if len(kps1) > 0 and len(kps2) > 0:
                        epi = np.full((len(kps1), len(kps2)), -1.0)
                        for kps1_candidate in range(len(kps1)):
                            for kps2_candidate in range(len(kps2)):
                                dist = line2linedist(cam1.Pos,
                                                     ray1[kps1_candidate],
                                                     cam2.Pos,
                                                     ray2[kps2_candidate])
                                if dist < self.max_epi_dist:
                                    epi[kps1_candidate,
                                        kps2_candidate] = \
                                            1 - dist / self.max_epi_dist

                        if self.normalize_edges:
                            row_factor = np.clip(epi.sum(1), 1.0, None)
                            col_factor = np.clip(epi.sum(0), 1.0, None)
                            for i in range(len(row_factor)):
                                epi[i] /= row_factor[i]
                            for j in range(len(col_factor)):
                                epi[:, j] /= col_factor[j]
                        self.m_epi_edges[kps_id][view1][view2] = epi
                        self.m_epi_edges[kps_id][view2][view1] = epi.T

    def _calculate_temp_edges(self):
        for kps_id in range(self.n_kps):
            for view in range(self.n_views):
                rays = self.m_kps_rays[view][kps_id]
                if len(self.last_multi_kps3d) > 0 and len(rays) > 0:
                    temp = np.full((len(self.last_multi_kps3d), len(rays)),
                                   -1.0)
                    for pid, person_id in enumerate(self.last_multi_kps3d):
                        limb = self.last_multi_kps3d[person_id]
                        if limb[3, kps_id] > 0:
                            for kps_candidate in range(len(rays)):
                                dist = point2linedist(limb[:, kps_id][:3],
                                                      self.cameras[view].Pos,
                                                      rays[kps_candidate])
                                if dist < self.max_temp_dist:
                                    temp[
                                        pid,
                                        kps_candidate] = \
                                            1 - dist / self.max_temp_dist

                    if self.normalize_edges:
                        row_factor = np.clip(temp.sum(1), 1.0, None)
                        col_factor = np.clip(temp.sum(0), 1.0, None)
                        for i in range(len(row_factor)):
                            temp[i] /= row_factor[i]
                        for j in range(len(col_factor)):
                            temp[:, j] /= col_factor[j]
                    self.m_temp_edges[kps_id][view] = temp

    def _calculate_bone_nodes(self):
        for paf_id in range(self.n_pafs):
            kps1, kps2 = self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]
            for view in range(self.n_views):
                self.m_bone_nodes[paf_id][view] = []
                for kps1_candidate in range(len(self.kps2d[view][kps1])):
                    for kps2_candidate in range(len(self.kps2d[view][kps2])):
                        if self.pafs[view][paf_id][kps1_candidate,
                                                   kps2_candidate] > 0:
                            self.m_bone_nodes[paf_id][view].append(
                                (kps1_candidate, kps2_candidate))

    def _calculate_bone_epi_edges(self):
        for paf_id in range(self.n_pafs):
            kps_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
            for view1 in range(self.n_views - 1):
                for view2 in range(view1 + 1, self.n_views):
                    nodes1 = self.m_bone_nodes[paf_id][view1]
                    nodes2 = self.m_bone_nodes[paf_id][view2]
                    epi = np.full((len(nodes1), len(nodes2)), -1.0)
                    for bone1_id in range(len(nodes1)):
                        for bone2_id in range(len(nodes2)):
                            node1 = nodes1[bone1_id]
                            node2 = nodes2[bone2_id]
                            epidist = np.zeros(2)
                            for i in range(2):
                                epidist[i] = self.m_epi_edges[
                                    kps_pair[i]][view1][view2][node1[i],
                                                               node2[i]]
                            if epidist.min() < 0:
                                continue
                            epi[bone1_id, bone2_id] = epidist.mean()
                    self.m_bone_epi_edges[paf_id][view1][view2] = epi
                    self.m_bone_epi_edges[paf_id][view2][view1] = epi.T

    def _calculate_bone_temp_edges(self):
        for paf_id in range(self.n_pafs):
            kps_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
            for view in range(self.n_views):
                nodes = self.m_bone_nodes[paf_id][view]
                temp = np.full((len(self.last_multi_kps3d), len(nodes)), -1.0)
                for pid in range(len(temp)):
                    for node_candidate in range(len(nodes)):
                        node = nodes[node_candidate]
                        tempdist = []
                        for i in range(2):
                            tempdist.append(self.m_temp_edges[kps_pair[i]]
                                            [view][pid][node[i]])
                        if min(tempdist) > 0:
                            temp[
                                pid,
                                node_candidate] = sum(tempdist) / len(tempdist)
                self.m_bone_temp_edges[paf_id][view] = temp
