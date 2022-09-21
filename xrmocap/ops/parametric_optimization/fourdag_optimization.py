import copy
import numpy as np

from xrmocap.ops.parametric_optimization.base_optimization import (
    BaseOptimization, )
from xrmocap.utils.fourdag_utils import (
    rodrigues,
    rodrigues_jacobi,
    skel_info,
    welsch,
)


class SkelInfo():

    def __init__(self, kps_convention) -> None:
        self.kps_convention = kps_convention
        self.boneLen = np.zeros(
            skel_info[self.kps_convention]['n_kps'] - 1, dtype=np.float32)
        self.boneCnt = np.zeros(
            skel_info[self.kps_convention]['n_kps'] - 1, dtype=np.float32)
        self.active = 0.0
        self.shapeFixed = False
        self.data = np.zeros(
            3 + skel_info[self.kps_convention]['n_kps'] * 3 +
            skel_info[self.kps_convention]['shape_size'],
            dtype=np.float32)

    def push_previous_bones(self, skel):
        for joint_id in range(1, skel_info[self.kps_convention]['n_kps']):
            prtIdx = skel_info[self.kps_convention]['joint_parent'][joint_id]
            if skel[3, joint_id] > 0 and skel[3, prtIdx] > 0:
                len = np.linalg.norm(skel[:, joint_id][:3] -
                                     skel[:, prtIdx][:3])
                self.boneLen[joint_id -
                             1] = (self.boneCnt[joint_id - 1] *
                                   self.boneLen[joint_id - 1] + len) / (
                                       self.boneCnt[joint_id - 1] + 1)
                self.boneCnt[joint_id - 1] += 1

    def get_trans(self):
        return self.data[0:0 + 3]

    def get_pose(self):
        return self.data[3:3 + skel_info[self.kps_convention]['n_kps'] * 3]

    def get_trans_pose(self):
        return self.data[:3 + skel_info[self.kps_convention]['n_kps'] * 3]

    def get_shape(self):
        return self.data[-skel_info[self.kps_convention]['shape_size']:]


class Term():

    def __init__(self):
        # joint 3d
        self.wJ3d = 0.
        self.j3dTarget = None

        # bone 3d
        self.wBone3d = 0.
        self.bone3dTarget = None

        # joint 2d
        self.wJ2d = 0.
        self.projs = None
        self.j2dTarget = None

        # temporal
        self.wTemporalTrans = 0.
        self.wTemporalPose = 0.
        self.wTemporalShape = 0.
        self.paramPrev = None

        # regular
        self.wRegularPose = 0.
        self.wRegularShape = 0.

        self.wSquareShape = 0.


class SkelSolver():

    def __init__(self, kps_convention) -> None:
        self.kps_convention = kps_convention
        self.m_joints = np.array(
            skel_info[self.kps_convention]['m_joints']).reshape(
                3, skel_info[self.kps_convention]['n_kps'])
        self.m_jShapeBlend = np.array(
            skel_info[self.kps_convention]['m_jShapeBlend']).reshape(
                skel_info[self.kps_convention]['n_kps'] * 3,
                skel_info[self.kps_convention]['shape_size'])
        self.m_boneShapeBlend = np.zeros(
            (3 * (skel_info[self.kps_convention]['n_kps'] - 1),
             skel_info[self.kps_convention]['shape_size']),
            dtype=np.float32)
        for jIdx in range(1, skel_info[self.kps_convention]['n_kps']):

            self.m_boneShapeBlend[3 * (jIdx - 1):3 * (jIdx - 1)+3] = self.m_jShapeBlend[3 * jIdx:3 * jIdx+3] \
            - self.m_jShapeBlend[3 * skel_info[self.kps_convention]['joint_parent'][jIdx]:3 * skel_info[self.kps_convention]['joint_parent'][jIdx]+3]

    def CalcJFinal_1(self, chainWarps):
        jFinal = np.zeros((3, int(chainWarps.shape[1] / 4)), dtype=np.float32)
        for jIdx in range(jFinal.shape[1]):
            jFinal[:,
                   jIdx] = (chainWarps[0:0 + 3,
                                       4 * jIdx + 3:4 * jIdx + 3 + 1]).reshape(
                                           (-1))
        return jFinal

    def CalcJFinal_2(self, param, _jCut=-1):

        jCut = _jCut if _jCut > 0 else self.m_joints.shape[1]
        jBlend = self.CalcJBlend(param)
        return self.CalcJFinal_1(
            self.CalcChainWarps(self.CalcNodeWarps(param, jBlend[:, :jCut])))

    def CalcJBlend(self, param):
        jOffset = np.matmul(self.m_jShapeBlend, param.get_shape())
        jBlend = self.m_joints + jOffset.reshape((self.m_joints.shape[1], 3)).T
        return jBlend

    def CalcNodeWarps(self, param, jBlend):
        nodeWarps = np.zeros((4, jBlend.shape[1] * 4), dtype=np.float32)
        for jIdx in range(jBlend.shape[1]):
            matrix = np.identity(4, dtype=np.float32)
            if jIdx == 0:
                matrix[:3,
                       -1:] = (jBlend[:, jIdx] + param.get_trans()).reshape(
                           (-1, 1))
            else:
                matrix[:3, -1:] = (jBlend[:, jIdx] - jBlend[:, skel_info[
                    self.kps_convention]['joint_parent'][jIdx]]).reshape(
                        (-1, 1))

            matrix[:3, :3] = rodrigues(param.get_pose()[3 * jIdx:3 * jIdx + 3])
            nodeWarps[:4, 4 * jIdx:4 * jIdx + 4] = matrix
        return nodeWarps

    def CalcChainWarps(self, nodeWarps):
        chainWarps = np.zeros((4, nodeWarps.shape[1]), dtype=np.float32)
        for jIdx in range(int(nodeWarps.shape[1] / 4)):
            if jIdx == 0:
                chainWarps[:, jIdx * 4:jIdx * 4 +
                           4] = nodeWarps[:, jIdx * 4:jIdx * 4 + 4]
            else:
                chainWarps[:, jIdx * 4:jIdx * 4 + 4] = np.matmul(
                    chainWarps[:,
                               skel_info[self.kps_convention]['joint_parent']
                               [jIdx] * 4:skel_info[self.kps_convention]
                               ['joint_parent'][jIdx] * 4 + 4],
                    nodeWarps[:, jIdx * 4:jIdx * 4 + 4])
        return chainWarps

    def AlignRT(self, term, param):
        # align root affine
        param.data[0:0 + 3] = term.j3dTarget[:, 0][:3] - self.m_joints[:, 0]

        def CalcAxes(xAxis, yAxis):
            axes = np.zeros((3, 3), dtype=np.float32)
            axes[:, 0] = xAxis / np.linalg.norm(xAxis)
            axes[:, 2] = np.cross(xAxis, yAxis) / np.linalg.norm(
                np.cross(xAxis, yAxis))
            axes[:, 1] = np.cross(axes[:, 2], axes[:, 0]) / np.linalg.norm(
                np.cross(axes[:, 2], axes[:, 0]))
            return axes

        mat =np.matmul(CalcAxes(term.j3dTarget[:,2][:3] - term.j3dTarget[:,1][:3],term.j3dTarget[:,3][:3] - term.j3dTarget[:,1][:3]), \
             (np.linalg.inv(CalcAxes(self.m_joints[:,2] - self.m_joints[:,1], self.m_joints[:,3] - self.m_joints[:,1]))))
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
        param.data[3:3 + skel_info[self.kps_convention]['n_kps'] *
                   3][:3] = angle * np.array([x, y, z], dtype=np.float32)

    def SolvePose(self,
                  term,
                  param,
                  maxIterTime,
                  hierarchy=False,
                  updateThresh=1e-4):
        jBlend = self.CalcJBlend(param)
        hierSize = max(skel_info[self.kps_convention]['hierarchy_map'])
        hier = 0 if hierarchy else hierSize
        jCut = 0
        while hier <= hierSize:
            while jCut < skel_info[self.kps_convention]['n_kps'] and skel_info[
                    self.kps_convention]['hierarchy_map'][jCut] <= hier:
                jCut += 1
            for iterTime in range(maxIterTime):
                nodeWarps = self.CalcNodeWarps(param, jBlend[:, :jCut])
                chainWarps = self.CalcChainWarps(nodeWarps)
                jFinal = self.CalcJFinal_1(chainWarps)
                jointJacobi = np.zeros((3 * jCut, 3 + 3 * jCut),
                                       dtype=np.float32)
                ATA = np.zeros((3 + 3 * jCut, 3 + 3 * jCut), dtype=np.float32)
                ATb = np.zeros((3 + 3 * jCut), dtype=np.float32)
                nodeWarpsJacobi = np.zeros((9, 3 * jCut), dtype=np.float32)
                for jIdx in range(jCut):
                    nodeWarpsJacobi[:,
                                    3 * jIdx:3 * jIdx + 3] = rodrigues_jacobi(
                                        param.get_pose()[3 * jIdx:3 * jIdx +
                                                         3]).T

                for djIdx in range(jCut):
                    jointJacobi[3 * djIdx:3 * djIdx + 3, :3] = np.identity(
                        3, dtype=np.float32)
                    for dAxis in range(3):
                        dChainWarps = np.zeros((4, 4 * jCut), dtype=np.float32)
                        valid = np.zeros(jCut, dtype=np.float32)
                        valid[djIdx] = 1
                        dChainWarps[:3, 4 * djIdx:4 * djIdx +
                                    3] = nodeWarpsJacobi[:, 3 * djIdx +
                                                         dAxis].copy().reshape(
                                                             (3, 3)).T
                        if djIdx != 0:
                            dChainWarps[:, 4 * djIdx:4 * djIdx + 4] = np.matmul(
                                chainWarps[:,
                                           4 * skel_info[self.kps_convention]
                                           ['joint_parent'][djIdx]:4 *
                                           skel_info[self.kps_convention]
                                           ['joint_parent'][djIdx] + 4],
                                dChainWarps[:, 4 * djIdx:4 * djIdx + 4])

                        for jIdx in range(djIdx + 1, jCut):
                            prtIdx = skel_info[
                                self.kps_convention]['joint_parent'][jIdx]
                            valid[jIdx] = valid[prtIdx]
                            if valid[jIdx]:
                                dChainWarps[:,
                                            4 * jIdx:4 * jIdx + 4] = np.matmul(
                                                dChainWarps[:, 4 *
                                                            prtIdx:4 * prtIdx +
                                                            4],
                                                nodeWarps[:, 4 *
                                                          jIdx:4 * jIdx + 4])
                                jointJacobi[jIdx * 3:jIdx * 3 + 3,
                                            3 + djIdx * 3 + dAxis:3 +
                                            djIdx * 3 + dAxis +
                                            1] = dChainWarps[0:0 + 3,
                                                             4 * jIdx +
                                                             3:4 * jIdx + 3 +
                                                             1]

                if term.wJ3d > 0:
                    for jIdx in range(jCut):
                        if term.j3dTarget[3, jIdx] > 0:
                            w = term.wJ3d * term.j3dTarget[3, jIdx]
                            jacobi = jointJacobi[3 * jIdx:3 * jIdx + 3]
                            ATA += w * np.matmul(jacobi.T, jacobi)
                            ATb += w * np.matmul(
                                jacobi.T,
                                (term.j3dTarget[0:0 + 3, jIdx:jIdx + 1] -
                                 jFinal[:, jIdx].reshape((-1, 1)))).reshape(-1)

                if term.wJ2d > 0:
                    for view in range(int(term.projs.shape[1] / 4)):
                        j2dTarget = term.j2dTarget[:, view * skel_info[
                            self.kps_convention]['n_kps']:view * skel_info[
                                self.kps_convention]['n_kps'] + skel_info[
                                    self.kps_convention]['n_kps']]
                        if sum(j2dTarget[2] > 0) > 0:
                            proj = term.projs[:, view * 4:view * 4 + 4]
                            for jIdx in range(jCut):
                                if j2dTarget[2, jIdx] > 0:
                                    abc = np.matmul(
                                        proj, np.append(jFinal[:, jIdx], 1))
                                    projJacobi = np.array(
                                        [
                                            1.0 / abc[2], 0.0, -abc[0] /
                                            (abc[2] * abc[2]), 0.0, 1.0 /
                                            abc[2], -abc[1] / (abc[2] * abc[2])
                                        ],
                                        dtype=np.float32).reshape((2, 3))
                                    projJacobi = np.matmul(
                                        projJacobi, proj[:, :3])

                                    w = term.wJ2d * j2dTarget[2, jIdx]
                                    jacobi = np.matmul(
                                        projJacobi,
                                        jointJacobi[3 * jIdx:3 * jIdx + 3])

                                    ATA += w * np.matmul(jacobi.T, jacobi)
                                    ATb += w * np.matmul(
                                        jacobi.T, j2dTarget[:2, jIdx:jIdx +
                                                            1].reshape(-1) -
                                        abc[:2] / abc[2])

                if term.wTemporalTrans > 0:
                    ATA[:3, :3] += term.wTemporalTrans * np.identity(
                        3, dtype=np.float32)
                    ATb[:3] += term.wTemporalTrans * (
                        term.paramPrev.get_trans() - param.get_trans())

                if term.wTemporalPose > 0:
                    ATA[-3 * jCut:,
                        -3 * jCut:] += term.wTemporalPose * np.identity(
                            3 * jCut, dtype=np.float32)
                    ATb[-3 * jCut:] += term.wTemporalPose * (
                        term.paramPrev.get_pose()[:3 * jCut] -
                        param.get_pose()[:3 * jCut])

                if term.wRegularPose > 0:
                    ATA += term.wRegularPose * np.identity(
                        3 + 3 * jCut, dtype=np.float32)

                delta = np.linalg.solve(ATA, ATb)
                param.data[:3 + skel_info[self.kps_convention]['n_kps'] *
                           3][:3 + 3 * jCut] += delta

                if np.linalg.norm(delta) < updateThresh:
                    break
            hier += 1

    def SolveShape(self, term, param, maxIterTime, updateThresh=1e-4):
        for iterTime in range(maxIterTime):
            # calc status
            jBlend = self.CalcJBlend(param)
            ATA = np.zeros((skel_info[self.kps_convention]['shape_size'],
                            skel_info[self.kps_convention]['shape_size']),
                           dtype=np.float32)
            ATb = np.zeros(
                skel_info[self.kps_convention]['shape_size'], dtype=np.float32)

            if term.wBone3d > 0:
                for jIdx in range(1, skel_info[self.kps_convention]['n_kps']):
                    if term.bone3dTarget[1, jIdx - 1] > 0:
                        w = term.wBone3d * term.bone3dTarget[1, jIdx - 1]
                        prtIdx = skel_info[
                            self.kps_convention]['joint_parent'][jIdx]
                        dir = jBlend[:, jIdx] - jBlend[:, prtIdx]
                        jacobi = self.m_boneShapeBlend[3 * (jIdx - 1):3 *
                                                       (jIdx - 1) + 3]
                        ATA += w * np.matmul(jacobi.T, jacobi)
                        ATb += w * np.matmul(
                            jacobi.T, term.bone3dTarget[0, jIdx - 1] *
                            (dir / np.linalg.norm(dir)) - dir)

            if term.wJ3d > 0 or term.wJ2d > 0:
                chainWarps = self.CalcChainWarps(
                    self.CalcNodeWarps(param, jBlend))
                jFinal = self.CalcJFinal_1(chainWarps)
                jointJacobi = np.zeros(
                    (3 * skel_info[self.kps_convention]['n_kps'],
                     skel_info[self.kps_convention]['shape_size']),
                    dtype=np.float32)
                for jIdx in range(skel_info[self.kps_convention]['n_kps']):
                    if jIdx == 0:
                        jointJacobi[3 * jIdx:3 * jIdx +
                                    3] = self.m_jShapeBlend[3 * jIdx:3 * jIdx +
                                                            3]
                    else:
                        prtIdx = skel_info[
                            self.kps_convention]['joint_parent'][jIdx]
                        jointJacobi[3 * jIdx:3 * jIdx+3]= jointJacobi[3 * prtIdx:3 * prtIdx+3] \
                            + chainWarps[0:3,4*prtIdx+3] * (self.m_jShapeBlend[3 * jIdx:3 * jIdx+3] - self.m_jShapeBlend[3 * prtIdx:3 * prtIdx+3])

                if term.wJ3d > 0:
                    for jIdx in range(skel_info[self.kps_convention]['n_kps']):
                        if term.j3dTarget[3, jIdx] > 0:
                            w = term.wJ3d * term.j3dTarget[3, jIdx]
                            jacobi = jointJacobi[3 * jIdx:3 * jIdx + 3]
                            ATA += w * np.matmul(jacobi.T, jacobi)
                            ATb += w * np.matmul(
                                jacobi.T, (term.j3dTarget[0 + 3, jIdx + 1] -
                                           jFinal[:, jIdx]))

                if term.wJ2d > 0:
                    for view in range(int(len(term.projs[0]) / 4)):
                        j2dTarget = term.j2dTarget[
                            view *
                            skel_info[self.kps_convention]['n_kps']:view *
                            skel_info[self.kps_convention]['n_kps'] +
                            skel_info[self.kps_convention]['n_kps']]
                        if sum(j2dTarget[2] > 0) > 0:
                            proj = term.projs[view * 4:view * 4 + 4]
                            for jIdx in range(
                                    skel_info[self.kps_convention]['n_kps']):
                                if j2dTarget[2, jIdx] > 0:
                                    abc = proj * np.append(jFinal[:, jIdx], 1)
                                    projJacobi = np.zeros((2, 3),
                                                          dtype=np.float32)
                                    projJacobi = np.array([1.0 / abc[2], 0.0, -abc[0] / (abc[2]*abc[2]), \
                                        0.0, 1.0 / abc[2], -abc[1] / (abc[2]*abc[2])], dtype=np.float32).reshape((2,3))
                                    projJacobi = projJacobi * proj[:, :3]

                                    w = term.wJ2d * j2dTarget[2, jIdx]
                                    jacobi = projJacobi * jointJacobi[3 *
                                                                      jIdx:3 *
                                                                      jIdx + 3]
                                    ATA += w * np.matmul(jacobi.T, jacobi)
                                    ATb += w * np.matmul(
                                        jacobi.T, j2dTarget[0 + 2, jIdx + 1] -
                                        abc[:2] / abc[2])

            if term.wTemporalShape > 0:
                ATA += term.wTemporalShape * np.identity(
                    skel_info[self.kps_convention]['shape_size'],
                    dtype=np.float32)
                ATb += term.wTemporalShape * (
                    term.paramPrev.get_shape() - param.get_shape())

            if term.wSquareShape > 0:
                ATA += term.wSquareShape * np.identity(
                    skel_info[self.kps_convention]['shape_size'],
                    dtype=np.float32)
                ATb -= term.wSquareShape * param.get_shape()

            if term.wRegularShape > 0:
                ATA += term.wRegularShape * np.identity(
                    skel_info[self.kps_convention]['shape_size'],
                    dtype=np.float32)

            delta = np.linalg.solve(ATA, ATb)
            param.data[-skel_info[self.kps_convention]['shape_size']:] += delta

            if np.linalg.norm(delta) < updateThresh:
                break


class FourDAGOptimization(BaseOptimization):

    def __init__(self,
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
        self.min_triangulate_cnt = min_triangulate_cnt
        self.init_active = init_active
        self.triangulate_thresh = triangulate_thresh

        self.projs = None
        self.m_skels = dict()
        self.m_skelInfos = []
        self.m_solver = SkelSolver(kps_convention)

        self.kps_convention = kps_convention

    def update(self, skels2d):
        prevCnt = len(self.m_skels)
        info_index = 0
        for pid, corr_id in enumerate(skels2d):
            if len(self.m_skels) != len(self.m_skelInfos):
                import pdb
                pdb.set_trace()
            if pid < prevCnt:
                info = self.m_skelInfos[info_index][1]
                info_id = self.m_skelInfos[info_index][0]
                skel = self.m_skels[info_id]
                active = min(
                    info.active + self.active_rate *
                    (2.0 * welsch(self.min_track_cnt,
                                  sum(skels2d[corr_id][2] > 0)) - 1.0), 1.0)
                if info.active < 0:
                    self.m_skels.pop(info_id)
                    self.m_skelInfos.pop(info_index)
                    continue
                else:
                    info_index += 1

                if not info.shapeFixed:
                    skel = self.triangulate_person(skels2d[corr_id])
                    if sum(skel[3] > 0) >= self.min_triangulate_cnt:
                        info.push_previous_bones(skel)
                        if min(info.boneCnt) >= self.bone_capacity:
                            info.push_previous_bones(skel)
                            shapeTerm = Term()
                            shapeTerm.bone3dTarget = np.row_stack(
                                (info.boneLen.T,
                                 np.ones(
                                     info.boneLen.shape[0], dtype=np.float32)))
                            shapeTerm.wBone3d = self.w_bone3d
                            shapeTerm.wSquareShape = self.w_square_shape
                            self.m_solver.SolveShape(shapeTerm, info,
                                                     self.shape_max_iter)

                            # align pose
                            poseTerm = Term()
                            poseTerm.j3dTarget = skel
                            poseTerm.wJ3d = self.w_joint3d
                            poseTerm.wRegularPose = self.w_regular_pose
                            self.m_solver.AlignRT(poseTerm, info)
                            self.m_solver.SolvePose(poseTerm, info,
                                                    self.pose_max_iter)

                            skel[:3] = self.m_solver.CalcJFinal_2(info)
                            info.shapeFixed = True
                    self.m_skels[info_id] = skel

                else:
                    # align pose
                    poseTerm = Term()
                    poseTerm.wJ2d = self.w_joint2d
                    poseTerm.projs = self.projs
                    poseTerm.j2dTarget = copy.deepcopy(skels2d[corr_id])

                    # filter single view correspondence
                    corrCnt = np.zeros(
                        skel_info[self.kps_convention]['n_kps'],
                        dtype=np.float32)
                    jConfidence = np.ones(
                        skel_info[self.kps_convention]['n_kps'],
                        dtype=np.float32)
                    for view in range(int(self.projs.shape[1] / 4)):
                        corrCnt += ((
                            poseTerm.
                            j2dTarget[:, view *
                                      skel_info[self.
                                                kps_convention]['n_kps']:view *
                                      skel_info[self.kps_convention]['n_kps'] +
                                      skel_info[self.kps_convention]['n_kps']]
                            [2].T > 0).astype(np.int))

                    for jIdx in range(skel_info[self.kps_convention]['n_kps']):
                        if corrCnt[jIdx] <= 1:
                            jConfidence[jIdx] = 0
                            for view in range(int(self.projs.shape[1] / 4)):
                                poseTerm.j2dTarget[:, view * skel_info[
                                    self.kps_convention]['n_kps'] + jIdx] = 0

                    poseTerm.wRegularPose = self.w_regular_pose
                    poseTerm.paramPrev = info
                    poseTerm.wTemporalTrans = self.w_temporal_trans
                    poseTerm.wTemporalPose = self.w_temporal_pose
                    self.m_solver.SolvePose(poseTerm, info, self.pose_max_iter)
                    skel[:3] = self.m_solver.CalcJFinal_2(info)
                    skel[3] = jConfidence.T
                    # update active
                    info.active = active
            else:
                skel = self.triangulate_person(skels2d[corr_id])
                # alloc new person
                if sum(skel[3] > 0) >= self.min_triangulate_cnt:
                    self.m_skelInfos.append(
                        (corr_id, SkelInfo(self.kps_convention)))
                    info = self.m_skelInfos[-1][1]
                    info.push_previous_bones(skel)
                    info.active = self.init_active
                    self.m_skels[corr_id] = skel
        return self.m_skels
