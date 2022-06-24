from mmhuman3d.core.conventions.keypoints_mapping.human_data import (  # noqa:F401,E501
    APPROXIMATE_MAP, APPROXIMATE_MAPPING_LIST, HUMAN_DATA, HUMAN_DATA_BODY,
    HUMAN_DATA_HEAD, HUMAN_DATA_HIP, HUMAN_DATA_LEFT_HAND, HUMAN_DATA_LIMBS,
    HUMAN_DATA_LIMBS_INDEX, HUMAN_DATA_PALETTE, HUMAN_DATA_PARTS,
    HUMAN_DATA_RIGHT_HAND, HUMAN_DATA_SHOULDER,
)

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
