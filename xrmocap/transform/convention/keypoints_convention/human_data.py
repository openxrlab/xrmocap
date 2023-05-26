# yapf: disable
from xrprimer.transform.convention.keypoints_convention.human_data import (
    HUMAN_DATA_PARTS,
)

# yapf: enable

HUMAN_DATA_LIMB_NAMES = {
    'left_ankle': {
        'left_knee': 'left_lower_leg'
    },
    'right_ankle': {
        'right_knee': 'right_lower_leg'
    },
    'left_shoulder': {
        'left_elbow': 'left_upperarm'
    },
    'right_shoulder': {
        'right_elbow': 'right_upperarm'
    },
    'left_elbow': {
        'left_wrist': 'left_forearm'
    },
    'right_elbow': {
        'right_wrist': 'right_forearm'
    },
    'left_hip_extra': {
        'left_knee': 'left_thigh'
    },
    'right_hip_extra': {
        'right_knee': 'right_thigh'
    },
}

HUMAN_DATA_FOOT = [
    'left_ankle', 'left_foot', 'left_heel', 'left_ankle_openpose',
    'left_bigtoe_openpose', 'left_smalltoe_openpose', 'left_toe_3dhp',
    'left_bigtoe', 'left_smalltoe', 'right_ankle', 'right_foot', 'right_heel',
    'right_ankle_openpose', 'right_bigtoe_openpose', 'right_smalltoe_openpose',
    'right_toe_3dhp', 'right_bigtoe', 'right_smalltoe'
]

HUMAN_DATA_PARTS['foot'] = HUMAN_DATA_FOOT
