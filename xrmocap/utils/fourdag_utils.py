import math
import numpy as np

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

def line2linedist(pa, raya, pb, rayb):
    if abs(np.vdot(raya, rayb)) < 1e-5:
        return point2linedist(pa, pb, raya)
    else:
        ve = np.cross(raya, rayb)
        ve  = ve/np.linalg.norm(ve)
        ve =  abs(np.vdot((pa-pb), ve))
        return ve

def point2linedist(pa, pb, ray):
    ve = np.cross(pa-pb, ray)
    return np.linalg.norm(ve)