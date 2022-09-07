import math
import numpy as np
import copy
import os
from xrmocap.ops.triangulation import m_shape

###
jointSize = 19
shapeSize=10
joint_parent = [-1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 11, 12, 14, 13]
hierarchyMap = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
###

def Welsch(c, x):
    x = x / c
    return 1 - math.exp(- x * x /2)

def Skew(vec):
    skew = np.zeros((3,3),dtype=np.float32)
    skew = np.array( [0, -vec[2], vec[1], \
        vec[2], 0, -vec[0], \
        -vec[1], vec[0], 0],dtype=np.float32).reshape((3, 3))
    return skew

def Rodrigues(vec):
    theta = np.linalg.norm(vec)
    I = np.identity(3,dtype=np.float32)
    if abs(theta) < 1e-5:
        return I
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        itheta = 1 / theta
        r = vec / theta
        return c * I + np.matmul((1 - c) * r.reshape((-1,1)), r.reshape((1, -1))) + s * Skew(r)
    
def RodriguesJacobi(vec):
    theta = np.linalg.norm(vec)
    dSkew = np.zeros((3,9),dtype=np.float32)
    dSkew[0, 5] = dSkew[1, 6] = dSkew[2, 1] = -1
    dSkew[0, 7] = dSkew[1, 2] = dSkew[2, 3] = 1
    if abs(theta) < 1e-5:
        return -dSkew
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        c1 = 1 - c
        itheta = 1 / theta
        r = vec / theta
        rrt = np.matmul(r.reshape((-1,1)), r.reshape((1,-1)))
        skew = Skew(r)
        I = np.identity(3,dtype=np.float32)
        drrt = np.array([r[0] + r[0], r[1], r[2], r[1], 0, 0, r[2], 0, 0,\
            0, r[0], 0, r[0], r[1] + r[1], r[2], 0, r[2], 0,\
            0, 0, r[0], 0, 0, r[1], r[0], r[1], r[2] + r[2]], dtype=np.float32).reshape((3,9))
        jaocbi = np.zeros((3,9),dtype=np.float32)
        a = np.zeros((5,1),dtype=np.float32)
        for i in range(3):
            a = np.array([ -s * r[i], (s - 2 * c1*itheta)*r[i], c1 * itheta, (c - s * itheta)*r[i], s * itheta], dtype=np.float32).reshape((5,1))
            for j in range(3):
                for k in range(3):
                    
                    jaocbi[i, k + k + k + j] = (a[0] * I[j, k] + a[1] * rrt[j, k] +\
                        a[2] * drrt[i, j + j + j + k] + a[3] * skew[j, k] +\
                        a[4] * dSkew[i, j + j + j + k])
        return jaocbi

class SkelInfo():
    def __init__(self) -> None:
        self.boneLen = np.zeros(jointSize - 1,dtype=np.float32)
        self.boneCnt = np.zeros(jointSize - 1,dtype=np.float32)
        self.active = 0.0
        self.shapeFixed = False
        self.data = np.zeros(3+jointSize*3+shapeSize,dtype=np.float32)

    def PushPrevBones(self, skel):
        for joint_id in range(1,jointSize):
            prtIdx = joint_parent[joint_id]
            if skel[3, joint_id] > 0 and skel[3, prtIdx] > 0:
                len = np.linalg.norm(skel[:,joint_id][:3] - skel[:,prtIdx][:3])
                self.boneLen[joint_id - 1] = (self.boneCnt[joint_id - 1] *  self.boneLen[joint_id - 1] + len) / (self.boneCnt[joint_id - 1] + 1)
                self.boneCnt[joint_id - 1] += 1
        
    def GetTrans(self):
        return self.data[0:0+3]

    def GetPose(self):
        return self.data[3:3+jointSize * 3]

    def GetTransPose(self):
        return self.data[:3+jointSize * 3]

    def GetShape(self):
        return self.data[-shapeSize:]

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
        
class Triangulator():

    def __init__(self):
        self.points = None
        self.projs = None
        self.convergent = False
        self.loss = 0
        self.pos = np.zeros(3, dtype=np.float32)

    def Solve(self,maxIterTime = 20, updateTolerance = 1e-4, regularTerm = 1e-4):
        self.convergent = False
        self.loss = 10e9
        self.pos = np.zeros(3, dtype=np.float32)

        if sum(self.points[2] > 0) < 2:
            return

        for iterTime in range(maxIterTime):
            if self.convergent:
                break
            ATA = regularTerm * np.identity(3,dtype=np.float32)
            ATb = np.zeros(3,dtype=np.float32)
            for view in range(self.points.shape[1]):

                if self.points[2, view] > 0: 
                    proj = self.projs[:,4 * view: 4 * view+ 4]
                    xyz = np.matmul(proj, np.append(self.pos, 1))
                    jacobi = np.zeros((2, 3),dtype=np.float32)
                    jacobi = np.array([1.0 / xyz[2], 0.0, -xyz[0] / (xyz[2]*xyz[2]),
                        0.0, 1.0 / xyz[2], -xyz[1] / (xyz[2]*xyz[2])],dtype=np.float32).reshape((2,3))
                    jacobi = np.matmul(jacobi, proj[:,:3])
                    w = self.points[2, view]
                    ATA += w *np.matmul(jacobi.T, jacobi)
                    ATb += w * np.matmul(jacobi.T,(self.points[:,view][:2] - xyz[:2]/xyz[2]))
                
            
            delta = np.linalg.solve(ATA, ATb)
            self.loss = np.linalg.norm(delta)
            if np.linalg.norm(delta) < updateTolerance:
                self.convergent = True
            else:
                self.pos += delta

class SkelSolver():
    def __init__(self) -> None:
        self.m_joints = np.array(m_shape.m_joints).reshape(3,jointSize)
        self.m_jShapeBlend = np.array(m_shape.m_jShapeBlend).reshape(jointSize*3, shapeSize)
        assert self.m_joints.shape[1] == jointSize and self.m_jShapeBlend.shape[0] == 3 * jointSize and self.m_jShapeBlend.shape[1] ==  shapeSize 
        self.m_boneShapeBlend= np.zeros((3 * (jointSize - 1), shapeSize), dtype=np.float32)
        for jIdx in range(1,jointSize):
        
            self.m_boneShapeBlend[3 * (jIdx - 1):3 * (jIdx - 1)+3] = self.m_jShapeBlend[3 * jIdx:3 * jIdx+3] \
            - self.m_jShapeBlend[3 * joint_parent[jIdx]:3 * joint_parent[jIdx]+3]


    def CalcJFinal_1(self,chainWarps):
        jFinal = np.zeros((3, int(chainWarps.shape[1] / 4)), dtype = np.float32)
        for jIdx in range(jFinal.shape[1]):
            jFinal[:,jIdx] = (chainWarps[0:0+3,4 * jIdx + 3:4 * jIdx + 3+1]).reshape((-1))
        return jFinal

    def CalcJFinal_2(self, param, _jCut=-1):
    
        jCut = _jCut if _jCut > 0 else self.m_joints.shape[1]
        jBlend = self.CalcJBlend(param)
        return self.CalcJFinal_1(self.CalcChainWarps(self.CalcNodeWarps(param, jBlend[:,:jCut])))
    


    def CalcJBlend(self, param):
        jOffset = np.matmul(self.m_jShapeBlend, param.GetShape())
        # jBlend = self.m_joints + Eigen::Map<const Eigen::Matrix3Xf>(jOffset.data(), 3, self.m_joints.shape[1])
        jBlend = self.m_joints + jOffset.reshape((self.m_joints.shape[1],3)).T

        return jBlend

    def CalcNodeWarps(self, param, jBlend):
        nodeWarps = np.zeros((4, jBlend.shape[1] * 4), dtype=np.float32)
        for jIdx in range(jBlend.shape[1]):
            matrix = np.identity(4,dtype=np.float32)
            if jIdx == 0:
                matrix[:3,-1:] = (jBlend[:,jIdx] + param.GetTrans()).reshape((-1,1))
            else:
                matrix[:3,-1:] = (jBlend[:,jIdx] - jBlend[:,joint_parent[jIdx]]).reshape((-1,1)) 

            matrix[:3,:3] = Rodrigues(param.GetPose()[3*jIdx:3*jIdx+3])
            nodeWarps[:4,4 * jIdx:4 * jIdx+4] = matrix
        # import pdb; pdb.set_trace()
        return nodeWarps

    def CalcChainWarps(self,nodeWarps):
        chainWarps = np.zeros((4, nodeWarps.shape[1]), dtype=np.float32)
        for jIdx in range(int(nodeWarps.shape[1]/4)):
            if jIdx == 0:
                chainWarps[:,jIdx * 4:jIdx * 4+4] = nodeWarps[:,jIdx * 4:jIdx * 4+4]
            else:
                chainWarps[:,jIdx * 4:jIdx * 4+4] = np.matmul(chainWarps[:,joint_parent[jIdx] * 4:joint_parent[jIdx] * 4+4], nodeWarps[:,jIdx * 4:jIdx * 4+4])
        return chainWarps

    def AlignRT(self,term,param):
        # align root affine
        param.data[0:0+3] = term.j3dTarget[:,0][:3] - self.m_joints[:,0]
        def CalcAxes(xAxis, yAxis):
            axes = np.zeros((3,3), dtype=np.float32)
            axes[:,0] = xAxis / np.linalg.norm(xAxis)
            axes[:,2] = np.cross(xAxis,yAxis) / np.linalg.norm(np.cross(xAxis,yAxis))
            axes[:,1] =  np.cross(axes[:,2],axes[:,0]) / np.linalg.norm(np.cross(axes[:,2],axes[:,0]))
            return axes
        
        # import pdb; pdb.set_trace()
        mat =np.matmul(CalcAxes(term.j3dTarget[:,2][:3] - term.j3dTarget[:,1][:3],term.j3dTarget[:,3][:3] - term.j3dTarget[:,1][:3]), \
             (np.linalg.inv(CalcAxes(self.m_joints[:,2] - self.m_joints[:,1], self.m_joints[:,3] - self.m_joints[:,1]))))
        # angleAxis(mat)
        angle = np.arccos(( mat[0,0] + mat[1,1] + mat[2,2] - 1)/2)
        x = (mat[2,1] - mat[1,2])/np.sqrt((mat[2,1] - mat[1,2])**2+(mat[0,2] - mat[2,0])**2+(mat[1,0] - mat[0,1])**2)
        y = (mat[0,2] - mat[2,0])/np.sqrt((mat[2,1] - mat[1,2])**2+(mat[0,2] - mat[2,0])**2+(mat[1,0] - mat[0,1])**2)
        z = (mat[1,0] - mat[0,1])/np.sqrt((mat[2,1] - mat[1,2])**2+(mat[0,2] - mat[2,0])**2+(mat[1,0] - mat[0,1])**2)
        param.data[3:3+jointSize * 3][:3] = angle * np.array([x,y,z], dtype=np.float32)

    def SolvePose(self, term, param, maxIterTime,hierarchy = False, updateThresh = 1e-4):
        
        jBlend = self.CalcJBlend(param)

        hierSize = max(hierarchyMap)
        hier = 0 if hierarchy else hierSize
        jCut = 0
        while hier <= hierSize:
            while jCut < jointSize and hierarchyMap[jCut] <= hier:
                jCut += 1
            for iterTime in range(maxIterTime):
                nodeWarps = self.CalcNodeWarps(param, jBlend[:,:jCut])
                chainWarps = self.CalcChainWarps(nodeWarps)
                jFinal = self.CalcJFinal_1(chainWarps)
                jointJacobi = np.zeros((3 * jCut, 3 + 3 * jCut),dtype=np.float32)
                ATA = np.zeros((3 + 3 * jCut, 3 + 3 * jCut),dtype=np.float32)
                ATb = np.zeros((3 + 3 * jCut),dtype=np.float32)
                nodeWarpsJacobi = np.zeros((9, 3 * jCut),dtype=np.float32)
                for jIdx in range(jCut):
                    nodeWarpsJacobi[:,3 * jIdx:3 * jIdx+3] = RodriguesJacobi(param.GetPose()[3*jIdx:3*jIdx+3]).T

                for djIdx in range(jCut):
                    jointJacobi[3 * djIdx:3 * djIdx+3,:3] = np.identity(3,dtype=np.float32)
                    for dAxis in range(3):
                        dChainWarps = np.zeros((4, 4 * jCut),dtype=np.float32)
                        valid = np.zeros(jCut,dtype=np.float32)
                        valid[djIdx] = 1
                        dChainWarps[:3,4*djIdx:4*djIdx+3] = nodeWarpsJacobi[:,3 * djIdx + dAxis].copy().reshape((3,3)).T
                        if djIdx != 0:
                            dChainWarps[:,4 * djIdx:4 * djIdx+4] = np.matmul(chainWarps[:,4 * joint_parent[djIdx]:4 * joint_parent[djIdx]+4], dChainWarps[:,4*djIdx:4*djIdx+4])

                        for jIdx in range(djIdx+1, jCut):
                            prtIdx = joint_parent[jIdx]
                            valid[jIdx] = valid[prtIdx]
                            if valid[jIdx]:
                                dChainWarps[:,4 * jIdx:4 * jIdx+4] = np.matmul(dChainWarps[:,4 * prtIdx:4 * prtIdx+4], nodeWarps[:,4 * jIdx:4 * jIdx+4])
                                jointJacobi[jIdx * 3:jIdx * 3+3,3 + djIdx * 3 + dAxis:3 + djIdx * 3 + dAxis+1] = dChainWarps[0:0+3,4 * jIdx + 3:4 * jIdx + 3+1]

                if term.wJ3d > 0: #0
                    for jIdx in range(jCut):
                        if term.j3dTarget[3, jIdx] > 0:
                            w = term.wJ3d * term.j3dTarget[3, jIdx]
                            jacobi = jointJacobi[3 * jIdx:3 * jIdx+3]
                            ATA += w * np.matmul(jacobi.T, jacobi)
                            ATb += w * np.matmul(jacobi.T, (term.j3dTarget[0:0+3,jIdx:jIdx+1] - jFinal[:,jIdx].reshape((-1,1)))).reshape(-1)
        

                if term.wJ2d > 0: #1e-5
                    for view in range(int(term.projs.shape[1]/4)):
                        j2dTarget = term.j2dTarget[:,view*jointSize:view*jointSize+jointSize]
                        if sum(j2dTarget[2] > 0) > 0 :
                            proj = term.projs[:,view * 4:view * 4+4]
                            for jIdx in range(jCut):
                                if j2dTarget[2, jIdx] > 0:
                                    abc = np.matmul(proj, np.append(jFinal[:,jIdx], 1))
                                    projJacobi = np.array([1.0 / abc[2], 0.0, -abc[0] / (abc[2]*abc[2]),0.0, 1.0 / abc[2], -abc[1] / (abc[2]*abc[2])], dtype=np.float32).reshape((2,3))
                                    projJacobi = np.matmul(projJacobi, proj[:,:3])

                                    w = term.wJ2d * j2dTarget[2, jIdx]
                                    jacobi = np.matmul(projJacobi, jointJacobi[3 * jIdx:3 * jIdx+3])
                                    
                                    ATA += w * np.matmul(jacobi.T, jacobi)
                                    ATb += w * np.matmul(jacobi.T, j2dTarget[:2,jIdx:jIdx+1].reshape(-1) - abc[:2] / abc[2])

                if term.wTemporalTrans > 0: #0.1
                    ATA[:3,:3] += term.wTemporalTrans * np.identity(3,dtype=np.float32)
                    ATb[:3] += term.wTemporalTrans * (term.paramPrev.GetTrans() - param.GetTrans()) #param prev
                

                if term.wTemporalPose > 0: #0.01
                    ATA[-3 * jCut:,-3 * jCut:] += term.wTemporalPose * np.identity(3 * jCut,dtype=np.float32)
                    ATb[-3 * jCut:] += term.wTemporalPose * (term.paramPrev.GetPose()[:3 * jCut]
                        - param.GetPose()[:3 * jCut])
                
                if term.wRegularPose > 0: #0.001
                    ATA += term.wRegularPose * np.identity(3 + 3 * jCut,dtype=np.float32)
                
                delta = np.linalg.solve(ATA, ATb)
                param.data[:3+jointSize * 3][:3 + 3 * jCut] += delta
            
                if np.linalg.norm(delta) < updateThresh:
                    break
            hier+=1

    def SolveShape(self,term, param, maxIterTime, updateThresh=1e-4):
        for iterTime in range(maxIterTime):
            # calc status
            jBlend = self.CalcJBlend(param)
            ATA = np.zeros((shapeSize, shapeSize),dtype=np.float32)
            ATb = np.zeros(shapeSize,dtype=np.float32)
            
            if term.wBone3d > 0:
                for jIdx in range(1, jointSize):
                    if term.bone3dTarget[1, jIdx - 1] > 0 :
                        w = term.wBone3d * term.bone3dTarget[1, jIdx - 1]
                        prtIdx = joint_parent[jIdx]
                        dir = jBlend[:,jIdx] - jBlend[:,prtIdx]
                        jacobi = self.m_boneShapeBlend[3 * (jIdx - 1): 3 * (jIdx - 1)+3]
                        ATA += w * np.matmul(jacobi.T, jacobi)
                        ATb += w * np.matmul(jacobi.T, term.bone3dTarget[0, jIdx - 1]*(dir/np.linalg.norm(dir)) - dir)
                        
            if term.wJ3d > 0 or term.wJ2d > 0:
                chainWarps = self.CalcChainWarps(self.CalcNodeWarps(param, jBlend))
                jFinal = self.CalcJFinal_1(chainWarps)
                jointJacobi = np.zeros((3 * jointSize, shapeSize),dtype=np.float32)
                for jIdx in range(jointSize):
                    if jIdx == 0:
                        jointJacobi[3 * jIdx:3 * jIdx+3] = self.m_jShapeBlend[3 * jIdx:3 * jIdx+3]
                    else:
                        prtIdx = joint_parent[jIdx]
                        jointJacobi[3 * jIdx:3 * jIdx+3]= jointJacobi[3 * prtIdx:3 * prtIdx+3] \
                            + chainWarps[0:3,4*prtIdx+3] * (self.m_jShapeBlend[3 * jIdx:3 * jIdx+3] - self.m_jShapeBlend[3 * prtIdx:3 * prtIdx+3])
                    
                if term.wJ3d > 0:
                    for jIdx in range(jointSize):
                        if term.j3dTarget[3, jIdx] > 0:
                            w = term.wJ3d * term.j3dTarget[3, jIdx]
                            jacobi = jointJacobi[3 * jIdx:3 * jIdx+3]
                            ATA += w * np.matmul(jacobi.T, jacobi)
                            ATb += w * np.matmul(jacobi.T, (term.j3dTarget[0+3,jIdx+1] - jFinal[:,jIdx]))
                
                if term.wJ2d > 0:
                    for view in range(int(len(term.projs[0])/4)):
                        j2dTarget = term.j2dTarget[view*jointSize:view*jointSize+jointSize]
                        
                        if sum(j2dTarget[2] > 0) > 0:
                            proj = term.projs[view * 4: view * 4+4]
                            for jIdx in range(jointSize):
                                if j2dTarget[2, jIdx] > 0:
                                    abc = proj * np.append(jFinal[:,jIdx], 1) 
                                    projJacobi = np.zeros((2,3), dtype=np.float32)
                                    projJacobi = np.array([1.0 / abc[2], 0.0, -abc[0] / (abc[2]*abc[2]), \
                                        0.0, 1.0 / abc[2], -abc[1] / (abc[2]*abc[2])], dtype=np.float32).reshape((2,3))
                                    projJacobi = projJacobi * proj[:,:3]

                                    w = term.wJ2d * j2dTarget[2, jIdx]
                                    jacobi = projJacobi * jointJacobi[3*jIdx:3*jIdx+3]
                                    ATA += w * np.matmul(jacobi.T, jacobi)
                                    ATb += w * np.matmul(jacobi.T,j2dTarget[0+2,jIdx+1] - abc[:2] / abc[2])
                                
            if term.wTemporalShape > 0:
                ATA += term.wTemporalShape * np.identity(shapeSize,dtype=np.float32)
                ATb += term.wTemporalShape * (term.paramPrev.GetShape() - param.GetShape())
            

            if term.wSquareShape > 0:
                ATA += term.wSquareShape * np.identity(shapeSize,dtype=np.float32)
                ATb -= term.wSquareShape * param.GetShape()
            
            if term.wRegularShape > 0:
                ATA += term.wRegularShape *np.identity(shapeSize,dtype=np.float32)
            
            delta = np.linalg.solve(ATA, ATb)
            param.data[-shapeSize:] += delta

            if np.linalg.norm(delta) < updateThresh:
                break

class FourDAGTriangulator():
    def __init__(self,
                 m_activeRate: float=0.1,
                 m_minTrackJCnt: int=5,
                 m_boneCapacity: int=100,
                 m_wBone3d: float=1.0,
                 m_wSquareShape: float=1e-2,
                 m_shapeMaxIter: int=5,
                 m_wJ3d: float=1.0,
                 m_wRegularPose: float=1e-3,
                 m_poseMaxIter: int=20,
                 m_wJ2d: float=1e-5,
                 m_wTemporalTrans: float=1e-1,
                 m_wTemporalPose: float=1e-2,
                 m_minTriangulateJCnt: int=15,
                 m_initActive: float=0.9,
                 m_triangulateThresh: float=0.05,
                 logger=None):
        
        self.m_activeRate = m_activeRate
        self.m_minTrackJCnt = m_minTrackJCnt
        self.m_boneCapacity = m_boneCapacity
        self.m_wBone3d = m_wBone3d
        self.m_wSquareShape = m_wSquareShape
        self.m_shapeMaxIter = m_shapeMaxIter
        self.m_wJ3d = m_wJ3d
        self.m_wRegularPose = m_wRegularPose
        self.m_poseMaxIter = m_poseMaxIter
        self.m_wJ2d = m_wJ2d
        self.m_wTemporalTrans = m_wTemporalTrans
        self.m_wTemporalPose = m_wTemporalPose
        self.m_minTriangulateJCnt = m_minTriangulateJCnt
        self.m_initActive = m_initActive
        self.m_triangulateThresh = m_triangulateThresh

        self.projs = None
        self.m_skels = dict()
        self.m_skelInfos = []
        self.m_solver = SkelSolver()

    def TriangulatePerson(self, skel2d):
        skel = np.zeros((4, jointSize),dtype=np.float32)
        triangulator = Triangulator()
        triangulator.projs = self.projs
        triangulator.points = np.zeros((3,int(self.projs.shape[1]/4)),dtype=np.float32)
        for jIdx in range(jointSize):
            for view in range(int(self.projs.shape[1] / 4)):
                skel_tmp = skel2d[:,view*jointSize:view*jointSize+jointSize]
                triangulator.points[:,view] = skel_tmp[:,jIdx]
            triangulator.Solve()
            if triangulator.loss < self.m_triangulateThresh:
                skel[:,jIdx] = np.append(triangulator.pos, 1)
        
        return skel

    def triangulate_w_filter(self, skels2d):
        prevCnt = len(self.m_skels)
        info_index = 0
        for pid, corr_id in enumerate(skels2d):
            # skel2dCorr = skels2d[corr_id]
            if len(self.m_skels) != len(self.m_skelInfos):
                import pdb; pdb.set_trace()
            if pid < prevCnt:
                info = self.m_skelInfos[info_index][1]
                info_id = self.m_skelInfos[info_index][0]
                skel = self.m_skels[info_id]
                active = min(info.active + self.m_activeRate * (2.0 * Welsch(self.m_minTrackJCnt, sum(skels2d[corr_id][2] > 0)) - 1.0 ), 1.0)
                if info.active < 0:
                    self.m_skels.pop(info_id)
                    self.m_skelInfos.pop(info_index)
                    # info_index += 1
                    continue
                else:
                    info_index += 1

                if not info.shapeFixed:
                    skel = self.TriangulatePerson(skels2d[corr_id])
                    if sum(skel[3] > 0) >= self.m_minTriangulateJCnt:
                        info.PushPrevBones(skel)
                        if min(info.boneCnt) >= self.m_boneCapacity:
                            info.PushPrevBones(skel)
                            shapeTerm = Term()
                            shapeTerm.bone3dTarget = np.row_stack((info.boneLen.T, np.ones(info.boneLen.shape[0],dtype=np.float32)))
                            shapeTerm.wBone3d = self.m_wBone3d
                            shapeTerm.wSquareShape = self.m_wSquareShape
                            self.m_solver.SolveShape(shapeTerm, info, self.m_shapeMaxIter)

                            # align pose
                            poseTerm = Term()
                            poseTerm.j3dTarget = skel
                            poseTerm.wJ3d = self.m_wJ3d
                            poseTerm.wRegularPose = self.m_wRegularPose
                            self.m_solver.AlignRT(poseTerm, info)
                            self.m_solver.SolvePose(poseTerm, info, self.m_poseMaxIter)
                            
                            skel[:3] = self.m_solver.CalcJFinal_2(info)
                            info.shapeFixed = True
                    self.m_skels[info_id] = skel

                else:
                    # align pose
                    poseTerm = Term()
                    poseTerm.wJ2d = self.m_wJ2d
                    poseTerm.projs = self.projs
                    poseTerm.j2dTarget = copy.deepcopy(skels2d[corr_id])

                    # filter single view correspondence
                    corrCnt = np.zeros(jointSize,dtype=np.float32)
                    jConfidence = np.ones(jointSize,dtype=np.float32)
                    for view in range(int(self.projs.shape[1] / 4)):
                        corrCnt += ((poseTerm.j2dTarget[:,view * jointSize:view * jointSize+jointSize][2].T > 0).astype(np.int))

                    for jIdx in range(jointSize):
                        if corrCnt[jIdx] <= 1:
                            jConfidence[jIdx] = 0
                            for view in range(int(self.projs.shape[1] / 4)):
                                poseTerm.j2dTarget[:,view * jointSize + jIdx] = 0
                    
                    poseTerm.wRegularPose = self.m_wRegularPose
                    poseTerm.paramPrev = info
                    poseTerm.wTemporalTrans = self.m_wTemporalTrans
                    poseTerm.wTemporalPose = self.m_wTemporalPose
                    self.m_solver.SolvePose(poseTerm, info, self.m_poseMaxIter)
                    skel[:3] = self.m_solver.CalcJFinal_2(info)
                    skel[3] = jConfidence.T
                    # update active
                    info.active = active
            else:
                skel = self.TriangulatePerson(skels2d[corr_id])
                # alloc new person
                if sum(skel[3]> 0) >= self.m_minTriangulateJCnt:
                    self.m_skelInfos.append((corr_id, SkelInfo()))
                    info = self.m_skelInfos[-1][1]
                    info.PushPrevBones(skel)
                    info.active = self.m_initActive
                    self.m_skels[corr_id] = skel
        return self.m_skels
                

    def set_cameras(self, cameras):
        self.cameras = cameras
        self.projs = np.zeros((3, len(self.cameras) * 4))
        for view in range(len(self.cameras)):
            self.projs[:,4*view:4*view + 4] = self.cameras[view].cvProj

    def triangulate_wo_filter(self, skels2d):
        skelIter = iter(self.m_skels.copy())
        prevCnt = len(self.m_skels)
        for pIdx, corr_id in enumerate(skels2d):
            skel = self.TriangulatePerson(skels2d[corr_id])
            active = sum(skel[3]> 0) >= self.m_minTriangulateJCnt
            if pIdx < prevCnt:
                if active:
                    self.m_skels[next(skelIter)] = skel
                    # index += 1
                else:
                    self.m_skels.pop(next(skelIter))
            
            elif active:
                self.m_skels[corr_id] = skel
        
        return self.m_skels