import math
import numpy as np

skel_info = dict(
    fourdag_19 = dict(
        hierarchy_map = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        paf_dict = [[1, 2, 7,  0, 0, 3, 8,  1, 5,  11, 5, 1, 6,  12, 6, 1,  14, 13],
                    [0, 7, 13, 2, 3, 8, 14, 5, 11, 15, 9, 6, 12, 16, 10, 4, 17, 18]],
    )
    
)

all_paf_mapping = dict(
    OPENPOSE_25 = dict(
        fourdag_19 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,23],
        fourdag_17 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    )
)

def welsch(c, x):
    x = x / c
    return 1 - math.exp(- x * x /2)

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