import copy
import heapq
import logging
import numpy as np
from typing import Union

from xrmocap.utils.fourdag_utils import *
from xrmocap.utils.fourdag_utils import skel_info
from .base_matching import BaseMatching


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
        self.fstCnt = np.zeros(2, dtype=np.int8)
        self.secCnt = np.zeros(2, dtype=np.int8)
        self.vote = dict()

    def parse(self):
        self.fstCnt = np.zeros(2)
        self.secCnt = np.zeros(2)
        if len(self.vote) == 0:
            return

        _vote = copy.deepcopy(self.vote)
        for i in range(2):
            for index in range(2):
                person_id = max(_vote, key=lambda x: _vote[x][index])

                if i == 0:
                    self.fst[index] = person_id
                    self.fstCnt[index] = _vote[person_id][index]
                else:
                    self.sec[index] = person_id
                    self.secCnt[index] = _vote[person_id][index]
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

    def calcRay(self, uv):
        var = -self.c_Rt_Ki.dot(np.append(uv, 1).T)
        return var / np.linalg.norm(var)


class FourDAGMatching(BaseMatching):

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
                 min_asgn_cnt: int = 5,
                 normalize_edges: bool = True,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """

        Args:

            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(logger=logger)
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
        self.min_asgn_cnt = min_asgn_cnt
        self.normalize_edges = normalize_edges

        self.paf_dict = skel_info[kps_convention]['paf_dict']
        self.hierarchy_map = skel_info[kps_convention]['hierarchy_map']

        self.m_pafHier = np.zeros(self.n_pafs)
        for paf_id in range(self.n_pafs):
            self.m_pafHier[paf_id] = min(
                self.hierarchy_map[self.paf_dict[0][paf_id]],
                self.hierarchy_map[self.paf_dict[1][paf_id]])
        self.m_pafHierSize = self.m_pafHier.max()

        self.m_joint2paf = {i: [] for i in range(self.n_kps)}
        for paf_id in range(self.n_pafs):
            joint_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
            self.m_joint2paf[joint_pair[0]].append(paf_id)
            self.m_joint2paf[joint_pair[1]].append(paf_id)

        self.m_epiEdges = {
            i: {
                j: {k: -1
                    for k in range(self.n_views)}
                for j in range(self.n_views)
            }
            for i in range(self.n_kps)
        }
        self.m_tempEdges = {
            i: {j: -1
                for j in range(self.n_views)}
            for i in range(self.n_kps)
        }
        self.m_jointRays = {
            i: {j: []
                for j in range(self.n_kps)}
            for i in range(self.n_views)
        }
        self.m_boneNodes = {
            i: {j: []
                for j in range(self.n_views)}
            for i in range(self.n_pafs)
        }
        self.m_boneEpiEdges = {
            i: {
                j: {k: []
                    for k in range(self.n_views)}
                for j in range(self.n_views)
            }
            for i in range(self.n_pafs)
        }
        self.m_boneTempEdges = {
            i: {j: []
                for j in range(self.n_views)}
            for i in range(self.n_pafs)
        }
        self.m_assignMap = {
            i: {j: []
                for j in range(self.n_kps)}
            for i in range(self.n_views)
        }
        self.m_personsMap = dict()

        self.cameras = []
        self.kps2d_paf = []
        self.last_multi_kps3d = dict()
        self.cliques = []

    def set_cameras(self, cameras_param):
        for view in range(len(cameras_param)):
            self.cameras.append(Camera(cameras_param[view]))

    def __call__(self, kps2d_paf, last_multi_kps3d=dict):
        """Match people id from different cameras.

        Args:

        Raises:

        Returns:
        """
        self.kps2d_paf = kps2d_paf
        self.last_multi_kps3d = last_multi_kps3d
        self.calculate_joint_rays()
        self.calculate_paf_edges()
        self.calculate_epi_edges()
        self.calculate_temp_edges()

        self.calculate_bone_nodes()
        self.calculate_bone_epi_edges()
        self.calculate_bone_temp_edges()

        self.initialize()
        self.enumerate_clques()
        while len(self.cliques) > 0:
            self.assign_top_clique()

        return self.m_personsMap

    def calculate_joint_rays(self):
        for view in range(self.n_views):
            cam = self.cameras[view]
            for joint_id in range(self.n_kps):
                self.m_jointRays[view][joint_id] = []
                joints = self.kps2d_paf[view]['joints'][joint_id]
                for joint_candidate in range(len(joints)):
                    self.m_jointRays[view][joint_id].append(
                        cam.calcRay(joints[joint_candidate][:2]))

    def calculate_paf_edges(self):
        if self.normalize_edges:
            for paf_id in range(self.n_pafs):
                for detection in self.kps2d_paf:
                    pafs = detection['pafs'][paf_id]
                    if np.sum(pafs) > 0:
                        row_factor = np.clip(pafs.sum(1), 1.0, None)
                        col_factor = np.clip(pafs.sum(0), 1.0, None)
                        for i in range(len(row_factor)):
                            pafs[i] /= row_factor[i]
                        for j in range(len(col_factor)):
                            pafs[:, j] /= col_factor[j]
                    detection['pafs'][paf_id] = pafs

    def calculate_epi_edges(self):
        for joint_id in range(self.n_kps):
            for view1 in range(self.n_views - 1):
                cam1 = self.cameras[view1]
                for view2 in range(view1 + 1, self.n_views):
                    cam2 = self.cameras[view2]
                    joint1 = self.kps2d_paf[view1]['joints'][joint_id]
                    joint2 = self.kps2d_paf[view2]['joints'][joint_id]
                    ray1 = self.m_jointRays[view1][joint_id]
                    ray2 = self.m_jointRays[view2][joint_id]

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
                        self.m_epiEdges[joint_id][view1][view2] = epi
                        self.m_epiEdges[joint_id][view2][view1] = epi.T

    def calculate_temp_edges(self):
        for joint_id in range(self.n_kps):
            for view in range(self.n_views):
                rays = self.m_jointRays[view][joint_id]
                if len(self.last_multi_kps3d) > 0 and len(rays) > 0:
                    temp = np.full((len(self.last_multi_kps3d), len(rays)),
                                   -1.0)
                    for pid, person_id in enumerate(self.last_multi_kps3d):
                        skel = self.last_multi_kps3d[person_id]
                        if skel[3, joint_id] > 0:
                            for joint_candidate in range(len(rays)):
                                dist = point2linedist(skel[:, joint_id][:3],
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
                    self.m_tempEdges[joint_id][view] = temp

    def calculate_bone_nodes(self):
        for paf_id in range(self.n_pafs):
            joint1, joint2 = self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]
            for view in range(self.n_views):
                self.m_boneNodes[paf_id][view] = []
                for joint1_candidate in range(
                        len(self.kps2d_paf[view]['joints'][joint1])):
                    for joint2_candidate in range(
                            len(self.kps2d_paf[view]['joints'][joint2])):
                        if self.kps2d_paf[view]['pafs'][paf_id][
                                joint1_candidate, joint2_candidate] > 0:
                            self.m_boneNodes[paf_id][view].append(
                                (joint1_candidate, joint2_candidate))

    def calculate_bone_epi_edges(self):
        for paf_id in range(self.n_pafs):
            joint_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
            for view1 in range(self.n_views - 1):
                for view2 in range(view1 + 1, self.n_views):
                    nodes1 = self.m_boneNodes[paf_id][view1]
                    nodes2 = self.m_boneNodes[paf_id][view2]
                    epi = np.full((len(nodes1), len(nodes2)), -1.0)
                    for bone1_id in range(len(nodes1)):
                        for bone2_id in range(len(nodes2)):
                            node1 = nodes1[bone1_id]
                            node2 = nodes2[bone2_id]
                            epidist = np.zeros(2)
                            for i in range(2):
                                epidist[i] = self.m_epiEdges[
                                    joint_pair[i]][view1][view2][node1[i],
                                                                 node2[i]]
                            if epidist.min() < 0:
                                continue
                            epi[bone1_id, bone2_id] = epidist.mean()
                    self.m_boneEpiEdges[paf_id][view1][view2] = epi
                    self.m_boneEpiEdges[paf_id][view2][view1] = epi.T

    def calculate_bone_temp_edges(self):
        for paf_id in range(self.n_pafs):
            joint_pair = [self.paf_dict[0][paf_id], self.paf_dict[1][paf_id]]
            for view in range(self.n_views):
                nodes = self.m_boneNodes[paf_id][view]
                temp = np.full((len(self.last_multi_kps3d), len(nodes)), -1.0)
                for pid in range(len(temp)):
                    for node_candidate in range(len(nodes)):
                        node = nodes[node_candidate]
                        tempdist = []
                        for i in range(2):
                            tempdist.append(self.m_tempEdges[joint_pair[i]]
                                            [view][pid][node[i]])
                        if min(tempdist) > 0:
                            temp[
                                pid,
                                node_candidate] = sum(tempdist) / len(tempdist)
                self.m_boneTempEdges[paf_id][view] = temp

    def initialize(self):
        for joint_id in range(self.n_kps):
            for view in range(self.n_views):
                self.m_assignMap[view][joint_id] = np.full(
                    len(self.kps2d_paf[view]['joints'][joint_id]), -1)

        self.m_personsMap = {}
        for pid in range(len(self.last_multi_kps3d)):
            self.m_personsMap[pid] = np.full((self.n_kps, self.n_views), -1)

    def enumerate_clques(self):
        tmpCliques = {i: [] for i in range(self.n_pafs)}
        for paf_id in range(self.n_pafs):
            nodes = self.m_boneNodes[paf_id]
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
                                clique.proposal[i] = available_node[i][i][
                                    pick[i]]
                        clique.score = self.calculate_clique_score(clique)
                        tmpCliques[paf_id].append(clique)
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
                                epiEdges = self.m_boneEpiEdges[paf_id][index -
                                                                       1][view]
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
                            tempEdge = self.m_boneTempEdges[paf_id][
                                self.n_views - 1]
                            bone1_id = available_node[self.n_views -
                                                      1][self.n_views -
                                                         1][pick[self.n_views -
                                                                 1]]
                            for pid in available_node[index - 1][self.n_views]:
                                if tempEdge[pid, bone1_id] > 0:
                                    available_node[index][self.n_views].append(
                                        pid)
                        else:
                            available_node[index][
                                self.n_views] = available_node[index - 1][
                                    self.n_views][:]

        for paf_id in range(self.n_pafs):
            self.cliques.extend(tmpCliques[paf_id])
        heapq.heapify(self.cliques)

    def assign_top_clique(self):
        clique = heapq.heappop(self.cliques)
        nodes = self.m_boneNodes[clique.paf_id]
        joint_pair = [
            self.paf_dict[0][clique.paf_id], self.paf_dict[1][clique.paf_id]
        ]
        if clique.proposal[self.n_views] != -1:
            person_id = clique.proposal[self.n_views]
            if self.checkCnt(clique, joint_pair, nodes, person_id) != -1:
                person = self.m_personsMap[person_id]
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] != -1:
                        node = nodes[view][clique.proposal[view]]
                        assign = (
                            self.m_assignMap[view][joint_pair[0]][node[0]],
                            self.m_assignMap[view][joint_pair[1]][node[1]])
                        if (assign[0] == -1 or assign[0] == person_id) and (
                                assign[1] == -1 or assign[1] == person_id):
                            for i in range(2):
                                person[joint_pair[i], view] = node[i]
                                self.m_assignMap[view][joint_pair[i]][
                                    node[i]] = person_id
                        else:
                            _proposal[view] = clique.proposal[view]
                self.m_personsMap[person_id] = person
                self.push_clique(clique.paf_id, _proposal[:])

            else:
                _proposal = clique.proposal
                _proposal[self.n_views] = -1
                self.push_clique(clique.paf_id, _proposal[:])

        else:
            voting = Voting()
            voting = self.clique2voting(clique, voting)
            voting.parse()

            if sum(voting.fstCnt) == 0:

                def allocFlag():
                    if sum(np.array(clique.proposal) >= 0) == 0:
                        return True
                    view_var = max(clique.proposal)
                    view = clique.proposal.index(view_var)
                    node = nodes[view][clique.proposal[view]]
                    person_candidate = []
                    for person_id in self.m_personsMap:

                        def checkCnt():
                            cnt = 0
                            for i in range(2):
                                _cnt = self.check_joint_compatibility(
                                    view, joint_pair[i], node[i], person_id)
                                if _cnt == -1:
                                    return -1
                                cnt += _cnt
                            return cnt

                        cntt = checkCnt()
                        if cntt >= self.min_check_cnt:
                            person_candidate.append([cntt, person_id])
                    if len(person_candidate) == 0:
                        return True
                    person_id = max(person_candidate)[1]
                    person = self.m_personsMap[person_id]
                    for i in range(2):
                        person[joint_pair[i], view] = node[i]
                        self.m_assignMap[view][joint_pair[i]][
                            node[i]] = person_id

                    self.m_personsMap[person_id] = person
                    return False

                # print('1. A & B not assigned yet')
                if allocFlag():
                    person = np.full((self.n_kps, self.n_views), -1)
                    if len(self.m_personsMap) == 0:
                        person_id = 0
                    else:
                        person_id = max(self.m_personsMap) + 1

                    for view in range(self.n_views):
                        if clique.proposal[view] >= 0:
                            node = nodes[view][clique.proposal[view]]
                            for i in range(2):
                                person[joint_pair[i], view] = node[i]
                                self.m_assignMap[view][joint_pair[i]][
                                    node[i]] = person_id
                    self.m_personsMap[person_id] = person

            elif min(voting.fstCnt) == 0:
                # print('2. A assigned but not B: Add B to person with A ')
                valid_id = 0 if voting.fstCnt[0] > 0 else 1
                master_id = voting.fst[valid_id]
                unassignj_id = joint_pair[1 - valid_id]
                person = self.m_personsMap[master_id]

                _proposal = [-1] * (self.n_views + 1)

                for view in range(self.n_views):
                    if clique.proposal[view] >= 0:
                        node = nodes[view][clique.proposal[view]]
                        unassignj_candidata = node[1 - valid_id]
                        assigned = self.m_assignMap[view][
                            joint_pair[valid_id]][node[valid_id]]
                        if assigned == master_id:
                            if person[
                                    unassignj_id,
                                    view] == -1 and self.check_joint_compatibility(
                                        view, unassignj_id,
                                        unassignj_candidata, master_id) >= 0:
                                person[unassignj_id,
                                       view] = unassignj_candidata
                                self.m_assignMap[view][unassignj_id][
                                    unassignj_candidata] = master_id
                            else:
                                continue

                        elif assigned == -1 and voting.fstCnt[valid_id] >= 2\
                             and voting.secCnt[valid_id] == 0 \
                             and (person[joint_pair[0], view] == -1 \
                             or person[joint_pair[0],view]==node[0]) \
                             and (person[joint_pair[1], view] == -1 \
                             or person[joint_pair[1],view]==node[1]):
                            if self.check_joint_compatibility(view, joint_pair[0], node[0],master_id) >= 0 \
                                and self.check_joint_compatibility(view, joint_pair[1], node[1],master_id) >= 0 :
                                for i in range(2):
                                    person[joint_pair[i], view] = node[i]
                                    self.m_assignMap[view][joint_pair[i]][
                                        node[i]] = master_id
                            else:
                                _proposal[view] = clique.proposal[view]
                        else:
                            _proposal[view] = clique.proposal[view]

                self.m_personsMap[master_id] = person
                if _proposal != clique.proposal:
                    self.push_clique(clique.paf_id, _proposal[:])

            elif voting.fst[0] == voting.fst[1]:
                # print('4. A & B already assigned to same person')
                master_id = voting.fst[0]
                person = self.m_personsMap[master_id]
                _proposal = [-1] * (self.n_views + 1)
                for view in range(self.n_views):
                    if clique.proposal[view] >= 0:
                        node = nodes[view][clique.proposal[view]]
                        assign_id = [
                            self.m_assignMap[view][joint_pair[0]][node[0]],
                            self.m_assignMap[view][joint_pair[1]][node[1]]
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
                                self.m_assignMap[view][unassignj_id][
                                    unassignj_candidata] = master_id
                            else:
                                _proposal[view] = clique.proposal[view]
                        elif max(assign_id) == -1 and sum(
                                voting.secCnt) == 0 and (
                                    person[joint_pair[0], view] == -1
                                    or person[joint_pair[0], view] == node[0]
                                ) and (person[joint_pair[1], view] == -1 or
                                       person[joint_pair[1], view] == node[1]):
                            for i in range(2):
                                person[joint_pair[i], view] = node[i]
                                self.m_assignMap[view][joint_pair[i]][
                                    node[i]] = master_id
                        else:
                            _proposal[view] = clique.proposal[view]

                    if _proposal != clique.proposal:
                        self.push_clique(clique.paf_id, _proposal[:])
                self.m_personsMap[master_id] = person

            else:
                # print('5. A & B already assigned to different people')
                for index in range(2):
                    while voting.secCnt[index] != 0:
                        master_id = min(voting.fst[index], voting.sec[index])
                        slave_id = max(voting.fst[index], voting.sec[index])
                        if slave_id > max(self.m_personsMap):
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
                            voting.secCnt[index] = voting.vote[iter][index]

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
                        master = self.m_personsMap[master_id]
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

    def calculate_clique_score(self, clique):
        scores = []
        for view1 in range(self.n_views - 1):
            if clique.proposal[view1] == -1:
                continue
            for view2 in range(view1 + 1, self.n_views):
                if clique.proposal[view2] == -1:
                    continue
                scores.append(self.m_boneEpiEdges[clique.paf_id][view1][view2][
                    clique.proposal[view1], clique.proposal[view2]])

        if len(scores) > 0:
            epiScore = sum(scores) / len(scores)
        else:
            epiScore = 1

        scores = []
        person_id = clique.proposal[self.n_views]
        if person_id != -1:
            for view in range(self.n_views):
                if clique.proposal[view] == -1:
                    continue
                scores.append(self.m_boneTempEdges[clique.paf_id][view][
                    person_id, clique.proposal[view]])

        if len(scores) > 0:
            tempScore = sum(scores) / len(scores)
        else:
            tempScore = 0

        scores = []
        for view in range(self.n_views):
            if clique.proposal[view] == -1:
                continue
            candidata_bone = self.m_boneNodes[clique.paf_id][view][
                clique.proposal[view]]
            scores.append(
                self.kps2d_paf[view]['pafs'][clique.paf_id][candidata_bone[0],
                                                            candidata_bone[1]])

        pafScore = sum(scores) / len(scores)
        var = sum(np.array(clique.proposal[:self.n_views]) >= 0)
        viewScore = welsch(self.c_view_cnt, var)
        hierScore = 1 - pow(self.m_pafHier[clique.paf_id] / self.m_pafHierSize,
                            4)
        return (self.w_epi * epiScore + self.w_temp * tempScore +
                self.w_paf * pafScore + self.w_view * viewScore +
                self.w_hier * hierScore) / (
                    self.w_epi + self.w_temp + self.w_paf + self.w_view +
                    self.w_hier)

    def checkCnt(self, clique, joint_pair, nodes, person_id):
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
        person = self.m_personsMap[pid]
        checkCnt = 0
        if person[joint_id][view] != -1 and person[joint_id][view] != candidate:
            return -1

        for paf_id in self.m_joint2paf[joint_id]:
            checkJIdx = self.paf_dict[0][paf_id] + self.paf_dict[1][
                paf_id] - joint_id
            if person[checkJIdx, view] == -1:
                continue
            joint_candidate1 = candidate
            joint_candidate2 = person[checkJIdx, view]
            if joint_id == self.paf_dict[1][paf_id]:
                joint_candidate1, joint_candidate2 = joint_candidate2, joint_candidate1

            if self.kps2d_paf[view]['pafs'][paf_id][joint_candidate1,
                                                    joint_candidate2] > 0:
                checkCnt = checkCnt + 1
            else:
                return -1

        for i in range(self.n_views):
            if i == view or person[joint_id, i] == -1:
                continue
            if self.m_epiEdges[joint_id][view][i][candidate,
                                                  int(person[joint_id,
                                                             i])] > 0:
                checkCnt = checkCnt + 1
            else:
                return -1
        return checkCnt

    def push_clique(self, paf_id, proposal):
        if max(proposal[:self.n_views]) == -1:
            return
        clique = Clique(paf_id, proposal)
        clique.score = self.calculate_clique_score(clique)
        heapq.heappush(self.cliques, clique)

    def check_person_compatibility_sview(self, master_id, slave_id, view):
        assert master_id < slave_id
        if slave_id < len(self.last_multi_kps3d):
            return -1
        checkCnt = 0
        master = self.m_personsMap[master_id]
        slave = self.m_personsMap[slave_id]

        for joint_id in range(self.n_kps):
            if master[joint_id,
                      view] != -1 and slave[joint_id, view] != -1 and master[
                          joint_id, view] != slave[joint_id, view]:
                return -1

        if master_id < len(self.last_multi_kps3d):
            for joint_id in range(self.n_kps):
                if slave[joint_id, view] != -1:
                    if self.m_tempEdges[joint_id][view][
                            master_id, slave[joint_id][view]] > 0:
                        checkCnt = checkCnt + 1
                    else:
                        return -1

        for paf_id in range(self.n_pafs):
            paf = self.kps2d_paf[view]['pafs'][paf_id]
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
                        checkCnt = checkCnt + 1
                    else:
                        return -1
        return checkCnt

    def check_person_compatibility(self, master_id, slave_id):
        assert master_id < slave_id
        if slave_id < len(self.last_multi_kps3d):
            return -1

        checkCnt = 0
        master = self.m_personsMap[master_id]
        slave = self.m_personsMap[slave_id]

        for view in range(self.n_views):
            _checkCnt = self.check_person_compatibility_sview(
                master_id, slave_id, view)
            if _checkCnt == -1:
                return -1
            else:
                checkCnt += _checkCnt

        for joint_id in range(self.n_kps):
            for view1 in range(self.n_views - 1):
                candidate1_id = master[joint_id, view1]
                if candidate1_id != -1:
                    for view2 in range(view1 + 1, self.n_views):
                        candidate2_id = slave[joint_id, view2]
                        if candidate2_id != -1:
                            if self.m_epiEdges[joint_id][view1][view2][
                                    candidate1_id, candidate2_id] > 0:
                                checkCnt += 1
                            else:
                                return -1
        return checkCnt

    def merge_person(self, master_id, slave_id):
        assert master_id < slave_id
        master = self.m_personsMap[master_id]
        slave = self.m_personsMap[slave_id]
        for view in range(self.n_views):
            for joint_id in range(self.n_kps):
                if slave[joint_id, view] != -1:
                    master[joint_id, view] = slave[joint_id, view]
                    self.m_assignMap[view][joint_id][slave[joint_id,
                                                           view]] = master_id

        self.m_personsMap[master_id] = master
        self.m_personsMap.pop(slave_id)

    def clique2voting(self, clique, voting):
        voting.vote = {}
        if len(self.m_personsMap) == 0:
            return voting

        for view in range(self.n_views):
            index = clique.proposal[view]
            if index != -1:
                node = self.m_boneNodes[clique.paf_id][view][index]
                for i in range(2):
                    assigned = self.m_assignMap[view][self.paf_dict[i][
                        clique.paf_id]][node[i]]
                    if assigned != -1:
                        if assigned not in voting.vote:
                            voting.vote[assigned] = np.zeros(2)

                        voting.vote[assigned][i] += 1
        return voting
