# yapf: disable
import copy
import numpy as np
from typing import Union

from xrmocap.ops.triangulation.builder import BaseTriangulator
from xrmocap.transform.keypoints3d.optim.fourdag_base_optimizer import (
    FourDAGBaseOptimizer,
)
from xrmocap.utils.fourdag_utils import (
    LIMB_INFO, rodrigues, rodrigues_jacobi, welsch,
)

# yapf: enable


class LimbInfo():

    def __init__(self, kps_convention) -> None:
        self.kps_convention = kps_convention
        self.boneLen = np.zeros(
            LIMB_INFO[self.kps_convention]['n_kps'] - 1, dtype=np.float32)
        self.boneCnt = np.zeros(
            LIMB_INFO[self.kps_convention]['n_kps'] - 1, dtype=np.float32)
        self.active = 0.0
        self.shape_fixed = False
        self.data = np.zeros(
            3 + LIMB_INFO[self.kps_convention]['n_kps'] * 3 +
            LIMB_INFO[self.kps_convention]['shape_size'],
            dtype=np.float32)

    def push_previous_bones(self, limb):
        for joint_id in range(1, LIMB_INFO[self.kps_convention]['n_kps']):
            prtIdx = LIMB_INFO[self.kps_convention]['joint_parent'][joint_id]
            if limb[3, joint_id] > 0 and limb[3, prtIdx] > 0:
                len = np.linalg.norm(limb[:, joint_id][:3] -
                                     limb[:, prtIdx][:3])
                self.boneLen[joint_id -
                             1] = (self.boneCnt[joint_id - 1] *
                                   self.boneLen[joint_id - 1] + len) / (
                                       self.boneCnt[joint_id - 1] + 1)
                self.boneCnt[joint_id - 1] += 1

    def get_trans(self):
        return self.data[:3]

    def get_pose(self):
        return self.data[3:3 + LIMB_INFO[self.kps_convention]['n_kps'] * 3]

    def get_trans_pose(self):
        return self.data[:3 + LIMB_INFO[self.kps_convention]['n_kps'] * 3]

    def get_shape(self):
        return self.data[-LIMB_INFO[self.kps_convention]['shape_size']:]


class Term():

    def __init__(self):
        # joint 3d
        self.w_joint3d = 0.
        self.j3d_target = None

        # bone 3d
        self.w_bone3d = 0.
        self.bone3d_target = None

        # joint 2d
        self.w_joint2d = 0.
        self.projs = None
        self.j2d_target = None

        # temporal
        self.w_temporal_trans = 0.
        self.w_temporal_pose = 0.
        self.w_temporal_shape = 0.
        self.paramPrev = None

        # regular
        self.w_regular_pose = 0.
        self.w_regular_shape = 0.
        self.w_square_shape = 0.


class LimbSolver():

    def __init__(self, kps_convention) -> None:
        self.kps_convention = kps_convention
        self.m_kps = np.array(LIMB_INFO[self.kps_convention]['m_kps']).reshape(
            3, LIMB_INFO[self.kps_convention]['n_kps'])
        self.shape_blend = np.array(
            LIMB_INFO[self.kps_convention]['shape_blend']).reshape(
                LIMB_INFO[self.kps_convention]['n_kps'] * 3,
                LIMB_INFO[self.kps_convention]['shape_size'])
        self.bone_shape_blend = np.zeros(
            (3 * (LIMB_INFO[self.kps_convention]['n_kps'] - 1),
             LIMB_INFO[self.kps_convention]['shape_size']),
            dtype=np.float32)
        for joint_id in range(1, LIMB_INFO[self.kps_convention]['n_kps']):
            self.bone_shape_blend[3 * (joint_id - 1):3 * (joint_id - 1)+3]\
                    = self.shape_blend[3 * joint_id:3 * joint_id+3] \
                    - self.shape_blend[3 * LIMB_INFO[
                        self.kps_convention]['joint_parent'][joint_id]:
                            3 * LIMB_INFO[
                            self.kps_convention]['joint_parent'][joint_id]+3]

    def cal_joint_final_1(self, chain_warps):
        joint_final = np.zeros((3, int(chain_warps.shape[1] / 4)),
                               dtype=np.float32)
        for joint_id in range(joint_final.shape[1]):
            joint_final[:, joint_id] = (chain_warps[0:0 + 3, 4 * joint_id +
                                                    3:4 * joint_id + 3 +
                                                    1]).reshape((-1))
        return joint_final

    def cal_joint_final_2(self, param, j_cut=-1):
        j_cut = j_cut if j_cut > 0 else self.m_kps.shape[1]
        joint_blend = self.cal_joint_blend(param)
        return self.cal_joint_final_1(
            self.cal_chain_warps(
                self.cal_node_warps(param, joint_blend[:, :j_cut])))

    def cal_joint_blend(self, param):
        jOffset = np.matmul(self.shape_blend, param.get_shape())
        joint_blend = self.m_kps + jOffset.reshape((self.m_kps.shape[1], 3)).T
        return joint_blend

    def cal_node_warps(self, param, joint_blend):
        node_warps = np.zeros((4, joint_blend.shape[1] * 4), dtype=np.float32)
        for joint_id in range(joint_blend.shape[1]):
            matrix = np.identity(4, dtype=np.float32)
            if joint_id == 0:
                matrix[:3, -1:] = (joint_blend[:, joint_id] +
                                   param.get_trans()).reshape((-1, 1))
            else:
                matrix[:3, -1:] = (
                    joint_blend[:, joint_id] -
                    joint_blend[:, LIMB_INFO[self.kps_convention]
                                ['joint_parent'][joint_id]]).reshape((-1, 1))

            matrix[:3, :3] = rodrigues(
                param.get_pose()[3 * joint_id:3 * joint_id + 3])
            node_warps[:4, 4 * joint_id:4 * joint_id + 4] = matrix
        return node_warps

    def cal_chain_warps(self, node_warps):
        chain_warps = np.zeros((4, node_warps.shape[1]), dtype=np.float32)
        for joint_id in range(int(node_warps.shape[1] / 4)):
            if joint_id == 0:
                chain_warps[:, joint_id * 4:joint_id * 4 +
                            4] = node_warps[:, joint_id * 4:joint_id * 4 + 4]
            else:
                chain_warps[:, joint_id * 4:joint_id * 4 + 4] = np.matmul(
                    chain_warps[:,
                                LIMB_INFO[self.kps_convention]['joint_parent']
                                [joint_id] * 4:LIMB_INFO[self.kps_convention]
                                ['joint_parent'][joint_id] * 4 + 4],
                    node_warps[:, joint_id * 4:joint_id * 4 + 4])
        return chain_warps

    def align_root_affine(self, term, param):
        # align root affine
        param.data[0:0 + 3] = term.j3d_target[:, 0][:3] - self.m_kps[:, 0]

        def cal_axes(x_axis, y_axis):
            axes = np.zeros((3, 3), dtype=np.float32)
            axes[:, 0] = x_axis / np.linalg.norm(x_axis)
            axes[:, 2] = np.cross(x_axis, y_axis) / np.linalg.norm(
                np.cross(x_axis, y_axis))
            axes[:, 1] = np.cross(axes[:, 2], axes[:, 0]) / np.linalg.norm(
                np.cross(axes[:, 2], axes[:, 0]))
            return axes

        mat = np.matmul(
            cal_axes(term.j3d_target[:, 2][:3] - term.j3d_target[:, 1][:3],
                     term.j3d_target[:, 3][:3] - term.j3d_target[:, 1][:3]),
            (np.linalg.inv(
                cal_axes(self.m_kps[:, 2] - self.m_kps[:, 1],
                         self.m_kps[:, 3] - self.m_kps[:, 1]))))
        angle = np.arccos((mat[0, 0] + mat[1, 1] + mat[2, 2] - 1) / 2)
        x = (mat[2, 1] - mat[1, 2]) / np.sqrt((mat[2, 1] - mat[1, 2])**2 +
                                              (mat[0, 2] - mat[2, 0])**2 +
                                              (mat[1, 0] - mat[0, 1])**2)
        y = (mat[0, 2] - mat[2, 0]) / np.sqrt((mat[2, 1] - mat[1, 2])**2 +
                                              (mat[0, 2] - mat[2, 0])**2 +
                                              (mat[1, 0] - mat[0, 1])**2)
        z = (mat[1, 0] - mat[0, 1]) / np.sqrt((mat[2, 1] - mat[1, 2])**2 +
                                              (mat[0, 2] - mat[2, 0])**2 +
                                              (mat[1, 0] - mat[0, 1])**2)
        param.data[3:3 + LIMB_INFO[self.kps_convention]['n_kps'] *
                   3][:3] = angle * np.array([x, y, z], dtype=np.float32)

    def solve_pose(self,
                   term,
                   param,
                   maxIter_time,
                   hierarchy=False,
                   update_thresh=1e-4):
        joint_blend = self.cal_joint_blend(param)
        hier_size = max(LIMB_INFO[self.kps_convention]['hierarchy_map'])
        hier = 0 if hierarchy else hier_size
        j_cut = 0
        while hier <= hier_size:
            while j_cut < LIMB_INFO[
                    self.kps_convention]['n_kps'] and LIMB_INFO[
                        self.kps_convention]['hierarchy_map'][j_cut] <= hier:
                j_cut += 1
            for iter_time in range(maxIter_time):
                node_warps = self.cal_node_warps(param, joint_blend[:, :j_cut])
                chain_warps = self.cal_chain_warps(node_warps)
                joint_final = self.cal_joint_final_1(chain_warps)
                joint_jacobi = np.zeros((3 * j_cut, 3 + 3 * j_cut),
                                        dtype=np.float32)
                ATA = np.zeros((3 + 3 * j_cut, 3 + 3 * j_cut),
                               dtype=np.float32)
                ATb = np.zeros((3 + 3 * j_cut), dtype=np.float32)
                node_warps_jacobi = np.zeros((9, 3 * j_cut), dtype=np.float32)
                for joint_id in range(j_cut):
                    node_warps_jacobi[:, 3 * joint_id:3 * joint_id +
                                      3] = rodrigues_jacobi(
                                          param.get_pose()[3 * joint_id:3 *
                                                           joint_id + 3]).T
                for d_jidx in range(j_cut):
                    joint_jacobi[3 * d_jidx:3 * d_jidx + 3, :3] = np.identity(
                        3, dtype=np.float32)
                    for dAxis in range(3):
                        d_chain_warps = np.zeros((4, 4 * j_cut),
                                                 dtype=np.float32)
                        valid = np.zeros(j_cut, dtype=np.float32)
                        valid[d_jidx] = 1
                        d_chain_warps[:3, 4 * d_jidx:4 * d_jidx +
                                      3] = node_warps_jacobi[:, 3 * d_jidx +
                                                             dAxis].copy(
                                                             ).reshape(
                                                                 (3, 3)).T
                        if d_jidx != 0:
                            d_chain_warps[:, 4 * d_jidx:4 * d_jidx +
                                          4] = np.matmul(
                                              chain_warps[:, 4 * LIMB_INFO[
                                                  self.kps_convention]
                                                          ['joint_parent']
                                                          [d_jidx]:4 *
                                                          LIMB_INFO[
                                                              self.
                                                              kps_convention]
                                                          ['joint_parent']
                                                          [d_jidx] + 4],
                                              d_chain_warps[:, 4 *
                                                            d_jidx:4 * d_jidx +
                                                            4])

                        for joint_id in range(d_jidx + 1, j_cut):
                            prtIdx = LIMB_INFO[
                                self.kps_convention]['joint_parent'][joint_id]
                            valid[joint_id] = valid[prtIdx]
                            if valid[joint_id]:
                                d_chain_warps[:, 4 * joint_id:4 * joint_id +
                                              4] = np.matmul(
                                                  d_chain_warps[:,
                                                                4 * prtIdx:4 *
                                                                prtIdx + 4],
                                                  node_warps[:,
                                                             4 * joint_id:4 *
                                                             joint_id + 4])
                                joint_jacobi[joint_id * 3:joint_id * 3 + 3,
                                             3 + d_jidx * 3 + dAxis:3 +
                                             d_jidx * 3 + dAxis +
                                             1] = d_chain_warps[0:0 + 3,
                                                                4 * joint_id +
                                                                3:4 *
                                                                joint_id + 3 +
                                                                1]
                if term.w_joint3d > 0:
                    for joint_id in range(j_cut):
                        if term.j3d_target[3, joint_id] > 0:
                            w = term.w_joint3d * term.j3d_target[3, joint_id]
                            jacobi = joint_jacobi[3 * joint_id:3 * joint_id +
                                                  3]
                            ATA += w * np.matmul(jacobi.T, jacobi)
                            ATb += w * np.matmul(jacobi.T, (
                                term.j3d_target[0:0 + 3, joint_id:joint_id + 1]
                                - joint_final[:, joint_id].reshape(
                                    (-1, 1)))).reshape(-1)

                if term.w_joint2d > 0:
                    for view in range(int(term.projs.shape[1] / 4)):
                        j2d_target = term.j2d_target[:, view * LIMB_INFO[
                            self.kps_convention]['n_kps']:view * LIMB_INFO[
                                self.kps_convention]['n_kps'] + LIMB_INFO[
                                    self.kps_convention]['n_kps']]
                        if sum(j2d_target[2] > 0) > 0:
                            proj = term.projs[:, view * 4:view * 4 + 4]
                            for joint_id in range(j_cut):
                                if j2d_target[2, joint_id] > 0:
                                    abc = np.matmul(
                                        proj,
                                        np.append(joint_final[:, joint_id], 1))
                                    proj_jacobi = np.array(
                                        [
                                            1.0 / abc[2], 0.0, -abc[0] /
                                            (abc[2] * abc[2]), 0.0, 1.0 /
                                            abc[2], -abc[1] / (abc[2] * abc[2])
                                        ],
                                        dtype=np.float32).reshape((2, 3))
                                    proj_jacobi = np.matmul(
                                        proj_jacobi, proj[:, :3])

                                    w = term.w_joint2d * j2d_target[2,
                                                                    joint_id]
                                    jacobi = np.matmul(
                                        proj_jacobi,
                                        joint_jacobi[3 *
                                                     joint_id:3 * joint_id +
                                                     3])

                                    ATA += w * np.matmul(jacobi.T, jacobi)
                                    ATb += w * np.matmul(
                                        jacobi.T,
                                        j2d_target[:2, joint_id:joint_id +
                                                   1].reshape(-1) -
                                        abc[:2] / abc[2])

                if term.w_temporal_trans > 0:
                    ATA[:3, :3] += term.w_temporal_trans * np.identity(
                        3, dtype=np.float32)
                    ATb[:3] += term.w_temporal_trans * (
                        term.paramPrev.get_trans() - param.get_trans())

                if term.w_temporal_pose > 0:
                    ATA[-3 * j_cut:,
                        -3 * j_cut:] += term.w_temporal_pose * np.identity(
                            3 * j_cut, dtype=np.float32)
                    ATb[-3 * j_cut:] += term.w_temporal_pose * (
                        term.paramPrev.get_pose()[:3 * j_cut] -
                        param.get_pose()[:3 * j_cut])

                if term.w_regular_pose > 0:
                    ATA += term.w_regular_pose * np.identity(
                        3 + 3 * j_cut, dtype=np.float32)

                delta = np.linalg.solve(ATA, ATb)
                param.data[:3 + LIMB_INFO[self.kps_convention]['n_kps'] *
                           3][:3 + 3 * j_cut] += delta

                if np.linalg.norm(delta) < update_thresh:
                    break
            hier += 1

    def solve_shape(self, term, param, maxIter_time, update_thresh=1e-4):
        for iter_time in range(maxIter_time):
            # calc status
            joint_blend = self.cal_joint_blend(param)
            ATA = np.zeros((LIMB_INFO[self.kps_convention]['shape_size'],
                            LIMB_INFO[self.kps_convention]['shape_size']),
                           dtype=np.float32)
            ATb = np.zeros(
                LIMB_INFO[self.kps_convention]['shape_size'], dtype=np.float32)

            if term.w_bone3d > 0:
                for joint_id in range(1,
                                      LIMB_INFO[self.kps_convention]['n_kps']):
                    if term.bone3d_target[1, joint_id - 1] > 0:
                        w = term.w_bone3d * term.bone3d_target[1, joint_id - 1]
                        prtIdx = LIMB_INFO[
                            self.kps_convention]['joint_parent'][joint_id]
                        dir = joint_blend[:, joint_id] - joint_blend[:, prtIdx]
                        jacobi = self.bone_shape_blend[3 * (joint_id - 1):3 *
                                                       (joint_id - 1) + 3]
                        ATA += w * np.matmul(jacobi.T, jacobi)
                        ATb += w * np.matmul(
                            jacobi.T, term.bone3d_target[0, joint_id - 1] *
                            (dir / np.linalg.norm(dir)) - dir)

            if term.w_joint3d > 0 or term.w_joint2d > 0:
                chain_warps = self.cal_chain_warps(
                    self.cal_node_warps(param, joint_blend))
                joint_final = self.cal_joint_final_1(chain_warps)
                joint_jacobi = np.zeros(
                    (3 * LIMB_INFO[self.kps_convention]['n_kps'],
                     LIMB_INFO[self.kps_convention]['shape_size']),
                    dtype=np.float32)
                for joint_id in range(LIMB_INFO[self.kps_convention]['n_kps']):
                    if joint_id == 0:
                        joint_jacobi[3 * joint_id:3 * joint_id +
                                     3] = self.shape_blend[3 * joint_id:3 *
                                                           joint_id + 3]
                    else:
                        prtIdx = LIMB_INFO[
                            self.kps_convention]['joint_parent'][joint_id]
                        joint_jacobi[3 * joint_id:3 * joint_id+3] =\
                            joint_jacobi[3 * prtIdx:3 * prtIdx+3] \
                            + chain_warps[:3, 4*prtIdx+3] * (
                                self.shape_blend[3 * joint_id:3 * joint_id+3] -
                                self.shape_blend[3 * prtIdx:3 * prtIdx+3])

                if term.w_joint3d > 0:
                    for joint_id in range(
                            LIMB_INFO[self.kps_convention]['n_kps']):
                        if term.j3d_target[3, joint_id] > 0:
                            w = term.w_joint3d * term.j3d_target[3, joint_id]
                            jacobi = joint_jacobi[3 * joint_id:3 * joint_id +
                                                  3]
                            ATA += w * np.matmul(jacobi.T, jacobi)
                            ATb += w * np.matmul(
                                jacobi.T,
                                (term.j3d_target[0 + 3, joint_id + 1] -
                                 joint_final[:, joint_id]))

                if term.w_joint2d > 0:
                    for view in range(int(len(term.projs[0]) / 4)):
                        j2d_target = term.j2d_target[
                            view *
                            LIMB_INFO[self.kps_convention]['n_kps']:view *
                            LIMB_INFO[self.kps_convention]['n_kps'] +
                            LIMB_INFO[self.kps_convention]['n_kps']]
                        if sum(j2d_target[2] > 0) > 0:
                            proj = term.projs[view * 4:view * 4 + 4]
                            for joint_id in range(
                                    LIMB_INFO[self.kps_convention]['n_kps']):
                                if j2d_target[2, joint_id] > 0:
                                    abc = proj * np.append(
                                        joint_final[:, joint_id], 1)
                                    proj_jacobi = np.zeros((2, 3),
                                                           dtype=np.float32)
                                    proj_jacobi = np.array(
                                        [
                                            1.0 / abc[2], 0.0, -abc[0] /
                                            (abc[2] * abc[2]), 0.0, 1.0 /
                                            abc[2], -abc[1] / (abc[2] * abc[2])
                                        ],
                                        dtype=np.float32).reshape((2, 3))
                                    proj_jacobi = proj_jacobi * proj[:, :3]

                                    w = term.w_joint2d * j2d_target[2,
                                                                    joint_id]
                                    jacobi = proj_jacobi * joint_jacobi[
                                        3 * joint_id:3 * joint_id + 3]
                                    ATA += w * np.matmul(jacobi.T, jacobi)
                                    ATb += w * np.matmul(
                                        jacobi.T,
                                        j2d_target[0 + 2, joint_id + 1] -
                                        abc[:2] / abc[2])

            if term.w_temporal_shape > 0:
                ATA += term.w_temporal_shape * np.identity(
                    LIMB_INFO[self.kps_convention]['shape_size'],
                    dtype=np.float32)
                ATb += term.w_temporal_shape * (
                    term.paramPrev.get_shape() - param.get_shape())

            if term.w_square_shape > 0:
                ATA += term.w_square_shape * np.identity(
                    LIMB_INFO[self.kps_convention]['shape_size'],
                    dtype=np.float32)
                ATb -= term.w_square_shape * param.get_shape()

            if term.w_regular_shape > 0:
                ATA += term.w_regular_shape * np.identity(
                    LIMB_INFO[self.kps_convention]['shape_size'],
                    dtype=np.float32)

            delta = np.linalg.solve(ATA, ATb)
            param.data[-LIMB_INFO[self.kps_convention]['shape_size']:] += delta

            if np.linalg.norm(delta) < update_thresh:
                break


class FourDAGOptimizer(FourDAGBaseOptimizer):

    def __init__(self,
                 triangulator: Union[None, dict, BaseTriangulator] = None,
                 active_rate: float = 0.1,
                 min_track_cnt: int = 5,
                 bone_capacity: int = 100,
                 w_bone3d: float = 1.0,
                 w_square_shape: float = 1e-2,
                 shape_max_iter: int = 5,
                 w_joint3d: float = 1.0,
                 w_regular_pose: float = 1e-3,
                 pose_max_iter: int = 20,
                 w_joint2d: float = 1e-5,
                 w_temporal_trans: float = 1e-1,
                 w_temporal_pose: float = 1e-2,
                 min_triangulate_cnt: int = 15,
                 init_active: float = 0.9,
                 triangulate_thresh: float = 0.05,
                 kps_convention: str = 'fourdag_19',
                 logger=None):
        """optimize with 2D projections loss, shape prior, temporal smoothing.

        Args:
            triangulator:
                triangulator to construct 3D keypoints
            active_rate (float):
                active value degression rate
            min_track_cnt (int):
                minimum track value
            bone_capacity (int):
                the minimum bone capacity to turn to optimization
            w_bone3d (float):
                weight for 3D kepoints loss to solve shape
            w_square_shape (float):
                weight for shape regulization loss
            shape_max_iter (int):
                maximal iteration to solve shape
            w_joint3d (float):
                weight for 3D kepoints loss to solve pose
            w_regular_pose (float):
                weight for pose regulization loss
            pose_max_iter (int):
                maximal iteration to solve pose
            w_joint2d (float):
                weight for 2D kepoints loss to solve pose
            w_temporal_trans (float):
                weight for temporal smoothing loss to solve pose
            w_temporal_pose (float):
                weight for temporal smoothing loss to solve pose
            min_triangulate_cnt (int):
                the minimum amount of 3D keypoints to be accepted
            init_active:
                initial weight for active value
            triangulate_thresh (float):
                the maximal triangulate loss to be accepted
            kps_convention (str):
                The name of keypoints convention.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """

        super().__init__(
            triangulator=triangulator,
            kps_convention=kps_convention,
            min_triangulate_cnt=min_triangulate_cnt,
            triangulate_thresh=triangulate_thresh,
            logger=logger)
        self.active_rate = active_rate
        self.min_track_cnt = min_track_cnt
        self.bone_capacity = bone_capacity
        self.w_bone3d = w_bone3d
        self.w_square_shape = w_square_shape
        self.shape_max_iter = shape_max_iter
        self.w_joint3d = w_joint3d
        self.w_regular_pose = w_regular_pose
        self.pose_max_iter = pose_max_iter
        self.w_joint2d = w_joint2d
        self.w_temporal_trans = w_temporal_trans
        self.w_temporal_pose = w_temporal_pose
        self.init_active = init_active

        self.limb_solver = LimbSolver(kps_convention)

    def update(self, limbs2d):
        for pid, corr_id in enumerate(limbs2d):
            if corr_id in self.trace_limbs:
                info = self.trace_limb_infos[corr_id]
                limb = self.trace_limbs[corr_id]
                active = min(
                    info.active + self.active_rate *
                    (2.0 * welsch(self.min_track_cnt,
                                  sum(limbs2d[corr_id][2] > 0)) - 1.0), 1.0)
                if info.active < 0:
                    self.trace_limbs.pop(corr_id)
                    self.trace_limb_infos.pop(corr_id)
                    continue

                if not info.shape_fixed:
                    limb = self.triangulate_person(limbs2d[corr_id])
                    if sum(limb[3] > 0) >= self.min_triangulate_cnt:
                        info.push_previous_bones(limb)
                        if min(info.boneCnt) >= self.bone_capacity:
                            info.push_previous_bones(limb)
                            shape_term = Term()
                            shape_term.bone3d_target = np.row_stack(
                                (info.boneLen.T,
                                 np.ones(
                                     info.boneLen.shape[0], dtype=np.float32)))
                            shape_term.w_bone3d = self.w_bone3d
                            shape_term.w_square_shape = self.w_square_shape
                            self.limb_solver.solve_shape(
                                shape_term, info, self.shape_max_iter)

                            # align pose
                            pose_term = Term()
                            pose_term.j3d_target = limb
                            pose_term.w_joint3d = self.w_joint3d
                            pose_term.w_regular_pose = self.w_regular_pose
                            self.limb_solver.align_root_affine(pose_term, info)
                            self.limb_solver.solve_pose(
                                pose_term, info, self.pose_max_iter)
                            limb[:3] = self.limb_solver.cal_joint_final_2(info)
                            info.shape_fixed = True
                    self.trace_limbs[corr_id] = limb

                else:
                    # align pose
                    pose_term = Term()
                    pose_term.w_joint2d = self.w_joint2d
                    pose_term.projs = self.projs
                    pose_term.j2d_target = copy.deepcopy(limbs2d[corr_id])
                    # filter single view correspondence
                    corr_cnt = np.zeros(
                        LIMB_INFO[self.kps_convention]['n_kps'],
                        dtype=np.float32)
                    j_confidence = np.ones(
                        LIMB_INFO[self.kps_convention]['n_kps'],
                        dtype=np.float32)
                    for view in range(int(self.projs.shape[1] / 4)):
                        corr_cnt += ((pose_term.j2d_target[:, view * LIMB_INFO[
                            self.kps_convention]['n_kps']:LIMB_INFO[
                                self.kps_convention]['n_kps'] *
                                                           (view + 1)][2].T >
                                      0).astype(np.int))
                    for joint_id in range(
                            LIMB_INFO[self.kps_convention]['n_kps']):
                        if corr_cnt[joint_id] <= 1:
                            j_confidence[joint_id] = 0
                            for view in range(int(self.projs.shape[1] / 4)):
                                pose_term.j2d_target[:, view * LIMB_INFO[
                                    self.kps_convention]['n_kps'] +
                                                     joint_id] = 0

                    pose_term.w_regular_pose = self.w_regular_pose
                    pose_term.paramPrev = info
                    pose_term.w_temporal_trans = self.w_temporal_trans
                    pose_term.w_temporal_pose = self.w_temporal_pose
                    self.limb_solver.solve_pose(pose_term, info,
                                                self.pose_max_iter)
                    limb[:3] = self.limb_solver.cal_joint_final_2(info)
                    limb[3] = j_confidence.T
                    # update active
                    info.active = active
            else:
                limb = self.triangulate_person(limbs2d[corr_id])
                # alloc new person
                if sum(limb[3] > 0) >= self.min_triangulate_cnt:
                    self.trace_limb_infos[corr_id] = LimbInfo(
                        self.kps_convention)
                    info = self.trace_limb_infos[corr_id]
                    info.push_previous_bones(limb)
                    info.active = self.init_active
                    self.trace_limbs[corr_id] = limb
        return self.trace_limbs
