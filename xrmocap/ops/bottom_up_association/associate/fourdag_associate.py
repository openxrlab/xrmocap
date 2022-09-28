import copy
import heapq
import logging
import numpy as np
from typing import Union

from xrmocap.utils.fourdag_utils import (
    LIMB_INFO,
    line2linedist,
    point2linedist,
    welsch,
)


class Clique():

    def __init__(self, paf_id, proposal, score=-1) -> None:
        self.paf_id = paf_id
        self.proposal = proposal
        self.score = score

    def __lt__(self, other):
        if self.score > other.score:
            return True
        else:
            return False


class Voting():

    def __init__(self) -> None:
        self.fst = np.zeros(2, dtype=np.int8)
        self.sec = np.zeros(2, dtype=np.int8)
        self.fst_cnt = np.zeros(2, dtype=np.int8)
        self.sec_cnt = np.zeros(2, dtype=np.int8)
        self.vote = dict()

    def parse(self):
        self.fst_cnt = np.zeros(2)
        self.sec_cnt = np.zeros(2)
        if len(self.vote) == 0:
            return

        _vote = copy.deepcopy(self.vote)
        for i in range(2):
            for index in range(2):
                person_id = max(_vote, key=lambda x: _vote[x][index])

                if i == 0:
                    self.fst[index] = person_id
                    self.fst_cnt[index] = _vote[person_id][index]
                else:
                    self.sec[index] = person_id
                    self.sec_cnt[index] = _vote[person_id][index]
                _vote[person_id][index] = 0


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


class FourDAGAssociate():

    def __init__(self,
                 kps_convention='fourdag_19',
                 n_views=5,
                 n_kps=19,
                 n_pafs=18,
                 max_epi_dist: float = 0.15,
                 max_temp_dist: float = 0.2,
                 w_epi: float = 2,
                 w_temp: float = 2,
                 w_view: float = 2,
                 w_paf: float = 1,
                 w_hier: float = 0.5,
                 c_view_cnt: float = 1.5,
                 min_check_cnt: int = 1,
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
            w_epi (float):
                clique score weight for epipolar distance
            w_temp (float):
                clique score weight for temporal tracking distance
            w_view (float):
                clique score weight for view number
            w_paf (float):
                clique score weight for paf edge
            w_hier (float):
                clique score weight for hierarchy
            c_view_cnt (float):
                maximal view number
            min_check_cnt (int):
                minimum check number
            normalize_edges (bool):
                indicator to normalize all edges
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = logger
        self.n_views = n_views
        self.n_kps = n_kps
        self.n_pafs = n_pafs
        self.max_epi_dist = max_epi_dist
        self.max_temp_dist = max_temp_dist
        self.w_epi = w_epi
        self.w_temp = w_temp
        self.w_view = w_view
        self.w_paf = w_paf
        self.w_hier = w_hier
        self.c_view_cnt = c_view_cnt
        self.min_check_cnt = min_check_cnt
        self.normalize_edges = normalize_edges

        self.paf_dict = LIMB_INFO[kps_convention]['paf_dict']
        self.hierarchy_map = LIMB_INFO[kps_convention]['hierarchy_map']

        self.m_paf_hier = np.zeros(self.n_pafs)
        for paf_id in range(self.n_pafs):
            self.m_paf_hier[paf_id] = min(
                self.hierarchy_map[self.paf_dict[0][paf_id]],
                self.hierarchy_map[self.paf_dict[1][paf_id]])
        self.m_paf_hier_size = self.m_paf_hier.max()

        self.m_joint2paf = {i: [] for i in range(self.n_kps)}
        for paf_id in range(self.n_pafs):
            joint_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
            self.m_joint2paf[joint_pair[0]].append(paf_id)
            self.m_joint2paf[joint_pair[1]].append(paf_id)

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
        self.m_joint_rays = {
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
        self.m_assign_map = {
            i: {j: []
                for j in range(self.n_kps)}
            for i in range(self.n_views)
        }
        self.m_persons_map = dict()

        self.cameras = []
        self.last_multi_kps3d = dict()
        self.cliques = []

    def set_cameras(self, cameras_param):
        for view in range(len(cameras_param)):
            self.cameras.append(Camera(cameras_param[view]))
        self.n_views = len(cameras_param)

    def __call__(self, kps2d, pafs, last_multi_kps3d=dict):
        """Match people id from different cameras.

        Args:

        Raises:

        Returns:
        """
        self.kps2d = kps2d
        self.pafs = pafs
        self.last_multi_kps3d = last_multi_kps3d
        self.construct_graph()
        self.solve_graph()

        return self.m_persons_map

    def construct_graph(self):
        self._calculate_joint_rays()
        self._calculate_paf_edges()
        self._calculate_epi_edges()
        self._calculate_temp_edges()

        self._calculate_bone_nodes()
        self._calculate_bone_epi_edges()
        self._calculate_bone_temp_edges()

    def solve_graph(self):
        self.initialize()
        self.enumerate_clques()
        while len(self.cliques) > 0:
            self.assign_top_clique()

    def _calculate_joint_rays(self):
        for view in range(self.n_views):
            cam = self.cameras[view]
            for joint_id in range(self.n_kps):
                self.m_joint_rays[view][joint_id] = []
                joints = self.kps2d[view][joint_id]
                for joint_candidate in range(len(joints)):
                    self.m_joint_rays[view][joint_id].append(
                        cam.cal_ray(joints[joint_candidate][:2]))

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
        for joint_id in range(self.n_kps):
            for view1 in range(self.n_views - 1):
                cam1 = self.cameras[view1]
                for view2 in range(view1 + 1, self.n_views):
                    cam2 = self.cameras[view2]
                    joint1 = self.kps2d[view1][joint_id]
                    joint2 = self.kps2d[view2][joint_id]
                    ray1 = self.m_joint_rays[view1][joint_id]
                    ray2 = self.m_joint_rays[view2][joint_id]

                    if len(joint1) > 0 and len(joint2) > 0:
                        epi = np.full((len(joint1), len(joint2)), -1.0)
                        for joint1_candidate in range(len(joint1)):
                            for joint2_candidate in range(len(joint2)):
                                dist = line2linedist(cam1.Pos,
                                                     ray1[joint1_candidate],
                                                     cam2.Pos,
                                                     ray2[joint2_candidate])
                                if dist < self.max_epi_dist:
                                    epi[joint1_candidate,
                                        joint2_candidate] = 1 - dist / self.max_epi_dist

                        if self.normalize_edges:
                            row_factor = np.clip(epi.sum(1), 1.0, None)
                            col_factor = np.clip(epi.sum(0), 1.0, None)
                            for i in range(len(row_factor)):
                                epi[i] /= row_factor[i]
                            for j in range(len(col_factor)):
                                epi[:, j] /= col_factor[j]
                        self.m_epi_edges[joint_id][view1][view2] = epi
                        self.m_epi_edges[joint_id][view2][view1] = epi.T

    def _calculate_temp_edges(self):
        for joint_id in range(self.n_kps):
            for view in range(self.n_views):
                rays = self.m_joint_rays[view][joint_id]
                if len(self.last_multi_kps3d) > 0 and len(rays) > 0:
                    temp = np.full((len(self.last_multi_kps3d), len(rays)),
                                   -1.0)
                    for pid, person_id in enumerate(self.last_multi_kps3d):
                        limb = self.last_multi_kps3d[person_id]
                        if limb[3, joint_id] > 0:
                            for joint_candidate in range(len(rays)):
                                dist = point2linedist(limb[:, joint_id][:3],
                                                      self.cameras[view].Pos,
                                                      rays[joint_candidate])
                                if dist < self.max_temp_dist:
                                    temp[
                                        pid,
                                        joint_candidate] = 1 - dist / self.max_temp_dist

                    if self.normalize_edges:
                        row_factor = np.clip(temp.sum(1), 1.0, None)
                        col_factor = np.clip(temp.sum(0), 1.0, None)
                        for i in range(len(row_factor)):
                            temp[i] /= row_factor[i]
                        for j in range(len(col_factor)):
                            temp[:, j] /= col_factor[j]
                    self.m_temp_edges[joint_id][view] = temp

    def _calculate_bone_nodes(self):
        for paf_id in range(self.n_pafs):
            joint1, joint2 = self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]
            for view in range(self.n_views):
                self.m_bone_nodes[paf_id][view] = []
                for joint1_candidate in range(len(self.kps2d[view][joint1])):
                    for joint2_candidate in range(
                            len(self.kps2d[view][joint2])):
                        if self.pafs[view][paf_id][joint1_candidate,
                                                   joint2_candidate] > 0:
                            self.m_bone_nodes[paf_id][view].append(
                                (joint1_candidate, joint2_candidate))

    def _calculate_bone_epi_edges(self):
        for paf_id in range(self.n_pafs):
            joint_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
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
                                    joint_pair[i]][view1][view2][node1[i],
                                                                 node2[i]]
                            if epidist.min() < 0:
                                continue
                            epi[bone1_id, bone2_id] = epidist.mean()
                    self.m_bone_epi_edges[paf_id][view1][view2] = epi
                    self.m_bone_epi_edges[paf_id][view2][view1] = epi.T

    def _calculate_bone_temp_edges(self):
        for paf_id in range(self.n_pafs):
            joint_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
            for view in range(self.n_views):
                nodes = self.m_bone_nodes[paf_id][view]
                temp = np.full((len(self.last_multi_kps3d), len(nodes)), -1.0)
                for pid in range(len(temp)):
                    for node_candidate in range(len(nodes)):
                        node = nodes[node_candidate]
                        tempdist = []
                        for i in range(2):
                            tempdist.append(self.m_temp_edges[joint_pair[i]]
                                            [view][pid][node[i]])
                        if min(tempdist) > 0:
                            temp[
                                pid,
                                node_candidate] = sum(tempdist) / len(tempdist)
                self.m_bone_temp_edges[paf_id][view] = temp

    def initialize(self):
        for joint_id in range(self.n_kps):
            for view in range(self.n_views):
                self.m_assign_map[view][joint_id] = np.full(
                    len(self.kps2d[view][joint_id]), -1)

        self.m_persons_map = {}
        for person_id in self.last_multi_kps3d:
            self.m_persons_map[person_id] = np.full((self.n_kps, self.n_views),
                                                    -1)

    def enumerate_clques(self):
        tmp_cliques = {i: [] for i in range(self.n_pafs)}
        for paf_id in range(self.n_pafs):
            nodes = self.m_bone_nodes[paf_id]
            pick = [-1] * (self.n_views + 1)
            available_node = {
                i: {j: []
                    for j in range(self.n_views + 1)}
                for i in range(self.n_views + 1)
            }

            # view_cnt = 0
            index = -1
            while True:
                if index >= 0 and pick[index] >= len(
                        available_node[index][index]):
                    pick[index] = -1
                    index = index - 1
                    if index < 0:
                        break
                    pick[index] += 1

                elif index == len(pick) - 1:
                    if sum(pick[:self.n_views]) != -self.n_views:
                        clique = Clique(paf_id, [-1] * len(pick))
                        for i in range(len(pick)):
                            if pick[i] != -1:
                                if i == len(pick) - 1:
                                    clique.proposal[i] = list(
                                        self.last_multi_kps3d.keys())[
                                            available_node[i][i][pick[i]]]
                                else:
                                    clique.proposal[i] = available_node[i][i][
                                        pick[i]]
                        clique.score = self.cal_clique_score(clique)
                        tmp_cliques[paf_id].append(clique)
                    pick[index] += 1

                else:
                    index += 1
                    if index == 0:
                        for view in range(self.n_views):
                            for bone in range(len(nodes[view])):
                                available_node[0][view].append(bone)
                        for pid in range(len(self.last_multi_kps3d)):
                            available_node[0][self.n_views].append(pid)

                    else:
                        if pick[index - 1] >= 0:
                            for view in range(index, self.n_views):
                                available_node[index][view] = []
                                epiEdges = self.m_bone_epi_edges[paf_id][
                                    index - 1][view]
                                bone1_id = available_node[index -
                                                          1][index -
                                                             1][pick[index -
                                                                     1]]
                                for bone2_id in available_node[index -
                                                               1][view]:
                                    if epiEdges[bone1_id, bone2_id] > 0:
                                        available_node[index][view].append(
                                            bone2_id)

                        else:
                            for view in range(index, self.n_views):
                                available_node[index][view] = available_node[
                                    index - 1][view][:]

                        if pick[self.n_views - 1] > 0:
                            available_node[index][self.n_views] = []
                            temp_edge = self.m_bone_temp_edges[paf_id][
                                self.n_views - 1]
                            bone1_id = available_node[self.n_views -
                                                      1][self.n_views -
                                                         1][pick[self.n_views -
                                                                 1]]
                            for pid in available_node[index - 1][self.n_views]:
                                if temp_edge[pid, bone1_id] > 0:
                                    available_node[index][self.n_views].append(
                                        pid)
                        else:
                            available_node[index][
                                self.n_views] = available_node[index - 1][
                                    self.n_views][:]

        for paf_id in range(self.n_pafs):
            self.cliques.extend(tmp_cliques[paf_id])
        heapq.heapify(self.cliques)

    def assign_top_clique(self):
        clique = heapq.heappop(self.cliques)
        nodes = self.m_bone_nodes[clique.paf_id]
        joint_pair = [
            self.paf_dict[0][clique.paf_id], self.paf_dict[1][clique.paf_id]
        ]
        if clique.proposal[self.n_views] != -1:
            person_id = clique.proposal[self.n_views]
            if self.check_cnt(clique, joint_pair, nodes, person_id) != -1:
                person = self.m_persons_map[person_id]
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] != -1:
                        node = nodes[view][clique.proposal[view]]
                        assign = (
                            self.m_assign_map[view][joint_pair[0]][node[0]],
                            self.m_assign_map[view][joint_pair[1]][node[1]])
                        if (assign[0] == -1 or assign[0] == person_id) and (
                                assign[1] == -1 or assign[1] == person_id):
                            for i in range(2):
                                person[joint_pair[i], view] = node[i]
                                self.m_assign_map[view][joint_pair[i]][
                                    node[i]] = person_id
                        else:
                            _proposal[view] = clique.proposal[view]
                self.m_persons_map[person_id] = person
                self.push_clique(clique.paf_id, _proposal[:])

            else:
                _proposal = clique.proposal
                _proposal[self.n_views] = -1
                self.push_clique(clique.paf_id, _proposal[:])

        else:
            voting = Voting()
            voting = self.clique2voting(clique, voting)
            voting.parse()

            if sum(voting.fst_cnt) == 0:

                def allocFlag():
                    if sum(np.array(clique.proposal) >= 0) == 0:
                        return True
                    view_var = max(clique.proposal)
                    view = clique.proposal.index(view_var)
                    node = nodes[view][clique.proposal[view]]
                    person_candidate = []
                    for person_id in self.m_persons_map:

                        def check_cnt():
                            cnt = 0
                            for i in range(2):
                                _cnt = self.check_joint_compatibility(
                                    view, joint_pair[i], node[i], person_id)
                                if _cnt == -1:
                                    return -1
                                cnt += _cnt
                            return cnt

                        cntt = check_cnt()
                        if cntt >= self.min_check_cnt:
                            person_candidate.append([cntt, person_id])
                    if len(person_candidate) == 0:
                        return True
                    person_id = max(person_candidate)[1]
                    person = self.m_persons_map[person_id]
                    for i in range(2):
                        person[joint_pair[i], view] = node[i]
                        self.m_assign_map[view][joint_pair[i]][
                            node[i]] = person_id

                    self.m_persons_map[person_id] = person
                    return False

                # ('1. A & B not assigned yet')
                if allocFlag():
                    person = np.full((self.n_kps, self.n_views), -1)
                    if len(self.m_persons_map) == 0:
                        person_id = 0
                    else:
                        person_id = max(self.m_persons_map) + 1

                    for view in range(self.n_views):
                        if clique.proposal[view] >= 0:
                            node = nodes[view][clique.proposal[view]]
                            for i in range(2):
                                person[joint_pair[i], view] = node[i]
                                self.m_assign_map[view][joint_pair[i]][
                                    node[i]] = person_id
                    self.m_persons_map[person_id] = person

            elif min(voting.fst_cnt) == 0:
                # ('2. A assigned but not B: Add B to person with A ')
                valid_id = 0 if voting.fst_cnt[0] > 0 else 1
                master_id = voting.fst[valid_id]
                unassignj_id = joint_pair[1 - valid_id]
                person = self.m_persons_map[master_id]
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] >= 0:
                        node = nodes[view][clique.proposal[view]]
                        unassignj_candidata = node[1 - valid_id]
                        assigned = self.m_assign_map[view][
                            joint_pair[valid_id]][node[valid_id]]
                        if assigned == master_id:
                            if person[
                                    unassignj_id,
                                    view] == -1 and self.check_joint_compatibility(
                                        view, unassignj_id,
                                        unassignj_candidata, master_id) >= 0:
                                person[unassignj_id,
                                       view] = unassignj_candidata
                                self.m_assign_map[view][unassignj_id][
                                    unassignj_candidata] = master_id
                            else:
                                continue

                        elif assigned == -1 and voting.fst_cnt[valid_id] >= 2\
                             and voting.sec_cnt[valid_id] == 0 \
                             and (person[joint_pair[0], view] == -1 \
                             or person[joint_pair[0],view]==node[0]) \
                             and (person[joint_pair[1], view] == -1 \
                             or person[joint_pair[1],view]==node[1]):
                            if self.check_joint_compatibility(view, joint_pair[0], node[0],master_id) >= 0 \
                                and self.check_joint_compatibility(view, joint_pair[1], node[1],master_id) >= 0 :
                                for i in range(2):
                                    person[joint_pair[i], view] = node[i]
                                    self.m_assign_map[view][joint_pair[i]][
                                        node[i]] = master_id
                            else:
                                _proposal[view] = clique.proposal[view]
                        else:
                            _proposal[view] = clique.proposal[view]

                self.m_persons_map[master_id] = person
                if _proposal != clique.proposal:
                    self.push_clique(clique.paf_id, _proposal[:])

            elif voting.fst[0] == voting.fst[1]:
                # ('4. A & B already assigned to same person')
                master_id = voting.fst[0]
                person = self.m_persons_map[master_id]
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] >= 0:
                        node = nodes[view][clique.proposal[view]]
                        assign_id = [
                            self.m_assign_map[view][joint_pair[0]][node[0]],
                            self.m_assign_map[view][joint_pair[1]][node[1]]
                        ]
                        if assign_id[0] == master_id and assign_id[
                                1] == master_id:
                            continue
                        elif self.check_joint_compatibility(
                                view, joint_pair[0], node[0], master_id
                        ) == -1 or self.check_joint_compatibility(
                                view, joint_pair[1], node[1], master_id) == -1:
                            _proposal[view] = clique.proposal[view]

                        elif (assign_id[0] == master_id and assign_id[1]
                              == -1) or (assign_id[0] == -1
                                         and assign_id[1] == master_id):
                            valid_id = 0 if assign_id[1] == -1 else 1
                            unassignj_id = joint_pair[1 - valid_id]
                            unassignj_candidata = node[1 - valid_id]
                            if person[unassignj_id, view] == -1 or person[
                                    unassignj_id, view] == unassignj_candidata:
                                person[unassignj_id,
                                       view] = unassignj_candidata
                                self.m_assign_map[view][unassignj_id][
                                    unassignj_candidata] = master_id
                            else:
                                _proposal[view] = clique.proposal[view]
                        elif max(assign_id) == -1 and sum(
                                voting.sec_cnt) == 0 and (
                                    person[joint_pair[0], view] == -1
                                    or person[joint_pair[0], view] == node[0]
                                ) and (person[joint_pair[1], view] == -1 or
                                       person[joint_pair[1], view] == node[1]):
                            for i in range(2):
                                person[joint_pair[i], view] = node[i]
                                self.m_assign_map[view][joint_pair[i]][
                                    node[i]] = master_id
                        else:
                            _proposal[view] = clique.proposal[view]

                    if _proposal != clique.proposal:
                        self.push_clique(clique.paf_id, _proposal[:])
                self.m_persons_map[master_id] = person

            else:
                # ('5. A & B already assigned to different people')
                for index in range(2):
                    while voting.sec_cnt[index] != 0:
                        master_id = min(voting.fst[index], voting.sec[index])
                        slave_id = max(voting.fst[index], voting.sec[index])
                        if slave_id > max(self.m_persons_map):
                            import pdb
                            pdb.set_trace()
                        if self.check_person_compatibility(
                                master_id, slave_id) >= 0:
                            self.merge_person(master_id, slave_id)
                            voting = self.clique2voting(clique, voting)
                            voting.parse()
                        else:
                            voting.vote[
                                voting.fst[index]][index] = voting.vote[
                                    voting.sec[index]][index] = 0
                            iter = max(
                                voting.vote,
                                key=lambda x: voting.vote[x][index])
                            voting.sec[index] = iter
                            voting.sec_cnt[index] = voting.vote[iter][index]

                if voting.fst[0] != voting.fst[1]:
                    conflict = [0] * self.n_views
                    master_id = min(voting.fst)
                    slave_id = max(voting.fst)
                    for view in range(self.n_views):
                        conflict[
                            view] = 1 if self.check_person_compatibility_sview(
                                master_id, slave_id, view) == -1 else 0

                    if sum(conflict) == 0:
                        self.merge_person(master_id, slave_id)
                        _proposal = [-1] * (self.n_views + 1)
                        master = self.m_persons_map[master_id]
                        for view in range(self.n_views):
                            if clique.proposal[view] >= 0:
                                if clique.proposal[view] >= len(nodes[view]):
                                    import pdb
                                    pdb.set_trace()
                                node = nodes[view][clique.proposal[view]]
                                if master[joint_pair[0],
                                          view] != node[0] or master[
                                              joint_pair[1], view] != node[1]:
                                    _proposal[view] = clique.proposal[view]
                        self.push_clique(clique.paf_id, _proposal[:])
                    else:
                        _proposal_pair = np.full((self.n_views + 1, 2), -1)
                        for i in range(len(conflict)):
                            _proposal_pair[i, conflict[i]] = clique.proposal[i]

                        if min(_proposal_pair[:, 0]) >= 0 and min(
                                _proposal_pair[:, 1]) >= 0:
                            self.push_clique(clique.paf_id,
                                             _proposal_pair[:, 0].copy())
                            self.push_clique(clique.paf_id,
                                             _proposal_pair[:, 1].copy())

                        elif sum(
                                np.array(clique.proposal[:self.n_views]) >= 0
                        ) > 1:
                            for i in range(len(conflict)):
                                _proposal = [-1] * (self.n_views + 1)
                                _proposal[i] = clique.proposal[i]
                                self.push_clique(clique.paf_id, _proposal[:])

    def cal_clique_score(self, clique):
        scores = []
        for view1 in range(self.n_views - 1):
            if clique.proposal[view1] == -1:
                continue
            for view2 in range(view1 + 1, self.n_views):
                if clique.proposal[view2] == -1:
                    continue
                scores.append(self.m_bone_epi_edges[clique.paf_id][view1]
                              [view2][clique.proposal[view1],
                                      clique.proposal[view2]])

        if len(scores) > 0:
            epi_score = sum(scores) / len(scores)
        else:
            epi_score = 1

        scores = []
        person_id = clique.proposal[self.n_views]
        if person_id != -1:
            for view in range(self.n_views):
                if clique.proposal[view] == -1:
                    continue
                scores.append(self.m_bone_temp_edges[clique.paf_id][view][
                    list(self.last_multi_kps3d.keys()).index(person_id),
                    clique.proposal[view]])

        if len(scores) > 0:
            temp_score = sum(scores) / len(scores)
        else:
            temp_score = 0

        scores = []
        for view in range(self.n_views):
            if clique.proposal[view] == -1:
                continue
            candidata_bone = self.m_bone_nodes[clique.paf_id][view][
                clique.proposal[view]]
            scores.append(self.pafs[view][clique.paf_id][candidata_bone[0],
                                                         candidata_bone[1]])

        paf_score = sum(scores) / len(scores)
        var = sum(np.array(clique.proposal[:self.n_views]) >= 0)
        view_score = welsch(self.c_view_cnt, var)
        hier_score = 1 - pow(
            self.m_paf_hier[clique.paf_id] / self.m_paf_hier_size, 4)
        return (self.w_epi * epi_score + self.w_temp * temp_score +
                self.w_paf * paf_score + self.w_view * view_score +
                self.w_hier * hier_score) / (
                    self.w_epi + self.w_temp + self.w_paf + self.w_view +
                    self.w_hier)

    def check_cnt(self, clique, joint_pair, nodes, person_id):
        cnt = 0
        for view in range(self.n_views):
            index = clique.proposal[view]
            if index != -1:
                for i in range(2):
                    _cnt = self.check_joint_compatibility(
                        view, joint_pair[i], nodes[view][index][i], person_id)
                    if _cnt == -1:
                        return -1
                    else:
                        cnt += _cnt
        return cnt

    def check_joint_compatibility(self, view, joint_id, candidate, pid):
        person = self.m_persons_map[pid]
        check_cnt = 0
        if person[joint_id][view] != -1 and person[joint_id][view] != candidate:
            return -1

        for paf_id in self.m_joint2paf[joint_id]:
            check_joint_id = self.paf_dict[0][paf_id] + self.paf_dict[1][
                paf_id] - joint_id
            if person[check_joint_id, view] == -1:
                continue
            joint_candidate1 = candidate
            joint_candidate2 = person[check_joint_id, view]
            if joint_id == self.paf_dict[1][paf_id]:
                joint_candidate1, joint_candidate2 = joint_candidate2, joint_candidate1

            if self.pafs[view][paf_id][joint_candidate1, joint_candidate2] > 0:
                check_cnt = check_cnt + 1
            else:
                return -1

        for i in range(self.n_views):
            if i == view or person[joint_id, i] == -1:
                continue
            if self.m_epi_edges[joint_id][view][i][candidate,
                                                   int(person[joint_id,
                                                              i])] > 0:
                check_cnt = check_cnt + 1
            else:
                return -1
        return check_cnt

    def push_clique(self, paf_id, proposal):
        if max(proposal[:self.n_views]) == -1:
            return
        clique = Clique(paf_id, proposal)
        clique.score = self.cal_clique_score(clique)
        heapq.heappush(self.cliques, clique)

    def check_person_compatibility_sview(self, master_id, slave_id, view):
        assert master_id < slave_id
        if slave_id < len(self.last_multi_kps3d):
            return -1
        check_cnt = 0
        master = self.m_persons_map[master_id]
        slave = self.m_persons_map[slave_id]

        for joint_id in range(self.n_kps):
            if master[joint_id,
                      view] != -1 and slave[joint_id, view] != -1 and master[
                          joint_id, view] != slave[joint_id, view]:
                return -1

        if master_id < len(self.last_multi_kps3d):
            for joint_id in range(self.n_kps):
                if slave[joint_id, view] != -1:
                    if self.m_temp_edges[joint_id][view][
                            master_id, slave[joint_id][view]] > 0:
                        check_cnt = check_cnt + 1
                    else:
                        return -1

        for paf_id in range(self.n_pafs):
            paf = self.pafs[view][paf_id]
            for candidate in [[
                    master[self.paf_dict[0][paf_id], view],
                    slave[self.paf_dict[1][paf_id], view]
            ],
                              [
                                  slave[self.paf_dict[0][paf_id], view],
                                  master[self.paf_dict[1][paf_id], view]
                              ]]:
                if min(candidate) >= 0:
                    if paf[candidate[0], candidate[1]] > 0:
                        check_cnt = check_cnt + 1
                    else:
                        return -1
        return check_cnt

    def check_person_compatibility(self, master_id, slave_id):
        assert master_id < slave_id
        if slave_id < len(self.last_multi_kps3d):
            return -1

        check_cnt = 0
        master = self.m_persons_map[master_id]
        slave = self.m_persons_map[slave_id]

        for view in range(self.n_views):
            _check_cnt = self.check_person_compatibility_sview(
                master_id, slave_id, view)
            if _check_cnt == -1:
                return -1
            else:
                check_cnt += _check_cnt

        for joint_id in range(self.n_kps):
            for view1 in range(self.n_views - 1):
                candidate1_id = master[joint_id, view1]
                if candidate1_id != -1:
                    for view2 in range(view1 + 1, self.n_views):
                        candidate2_id = slave[joint_id, view2]
                        if candidate2_id != -1:
                            if self.m_epi_edges[joint_id][view1][view2][
                                    candidate1_id, candidate2_id] > 0:
                                check_cnt += 1
                            else:
                                return -1
        return check_cnt

    def merge_person(self, master_id, slave_id):
        assert master_id < slave_id
        master = self.m_persons_map[master_id]
        slave = self.m_persons_map[slave_id]
        for view in range(self.n_views):
            for joint_id in range(self.n_kps):
                if slave[joint_id, view] != -1:
                    master[joint_id, view] = slave[joint_id, view]
                    self.m_assign_map[view][joint_id][slave[joint_id,
                                                            view]] = master_id

        self.m_persons_map[master_id] = master
        self.m_persons_map.pop(slave_id)

    def clique2voting(self, clique, voting):
        voting.vote = {}
        if len(self.m_persons_map) == 0:
            return voting

        for view in range(self.n_views):
            index = clique.proposal[view]
            if index != -1:
                node = self.m_bone_nodes[clique.paf_id][view][index]
                for i in range(2):
                    assigned = self.m_assign_map[view][self.paf_dict[i][
                        clique.paf_id]][node[i]]
                    if assigned != -1:
                        if assigned not in voting.vote:
                            voting.vote[assigned] = np.zeros(2)

                        voting.vote[assigned][i] += 1
        return voting
