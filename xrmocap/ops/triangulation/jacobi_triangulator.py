import numpy as np

class JacobiTriangulator():

    def __init__(self,logger=None):
        self.points = None
        self.projs = None
        self.convergent = False
        self.loss = 0
        self.pos = np.zeros(3, dtype=np.float32)

    def solve(self, maxIterTime=20, updateTolerance=1e-4, regularTerm=1e-4):
        self.convergent = False
        self.loss = 10e9
        self.pos = np.zeros(3, dtype=np.float32)

        if sum(self.points[2] > 0) < 2:
            return

        for iterTime in range(maxIterTime):
            if self.convergent:
                break
            ATA = regularTerm * np.identity(3, dtype=np.float32)
            ATb = np.zeros(3, dtype=np.float32)
            for view in range(self.points.shape[1]):
                if self.points[2, view] > 0:
                    proj = self.projs[:, 4 * view:4 * view + 4]
                    xyz = np.matmul(proj, np.append(self.pos, 1))
                    jacobi = np.zeros((2, 3), dtype=np.float32)
                    jacobi = np.array([
                        1.0 / xyz[2], 0.0, -xyz[0] / (xyz[2] * xyz[2]), 0.0,
                        1.0 / xyz[2], -xyz[1] / (xyz[2] * xyz[2])
                    ],
                                      dtype=np.float32).reshape((2, 3))
                    jacobi = np.matmul(jacobi, proj[:, :3])
                    w = self.points[2, view]
                    ATA += w * np.matmul(jacobi.T, jacobi)
                    ATb += w * np.matmul(
                        jacobi.T,
                        (self.points[:, view][:2] - xyz[:2] / xyz[2]))

            delta = np.linalg.solve(ATA, ATb)
            self.loss = np.linalg.norm(delta)
            if np.linalg.norm(delta) < updateTolerance:
                self.convergent = True
            else:
                self.pos += delta

