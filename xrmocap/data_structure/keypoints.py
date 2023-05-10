# yapf: disable
import logging
import numpy as np
import torch
from typing import Union
from xrprimer.data_structure import Keypoints as XRPrimerKeypoints

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# yapf: enable


class Keypoints(XRPrimerKeypoints):
    deprecation_warned = False
    """A class for multi-frame, multi-person keypoints data, based on python
    dict.

    keypoints, mask and convention are the three necessary keys, and we advise
    you to just call Keypoints(). If you'd like to set them manually, it is
    recommended to obey the following turn: convention -> keypoints -> mask.
    """

    def __init__(self,
                 src_dict: dict = None,
                 dtype: Literal['torch', 'numpy', 'auto'] = 'auto',
                 kps: Union[np.ndarray, torch.Tensor, None] = None,
                 mask: Union[np.ndarray, torch.Tensor, None] = None,
                 convention: Union[str, None] = None,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Construct a Keypoints instance with pre-set values. If any of kps,
        mask, convention is provided, it will override the item in src_dict.

        Args:
            src_dict (dict, optional):
                A dict with items in Keypoints fashion.
                Defaults to None.
            dtype (Literal['torch', 'numpy', 'auto'], optional):
                The data type of this Keypoints instance, values will
                be converted to the certain dtype when setting. If
                dtype==auto, it be changed the first time set_keypoints()
                is called, and never changes.
                Defaults to 'auto'.
            kps (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for keypoints,
                kps2d in shape [n_frame, n_person, n_kps, 3],
                kps3d in shape [n_frame, n_person, n_kps, 4].
                Shape [n_kps, 3 or 4] is also accepted, unsqueezed
                automatically. Defaults to None.
            mask (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for keypoint mask,
                in shape [n_frame, n_person, n_kps],
                in dtype uint8.
                Shape [n_kps, ] is also accepted, unsqueezed
                automatically. Defaults to None.
            convention (str, optional):
                Convention name of the keypoints,
                can be found in KEYPOINTS_FACTORY.
                Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        XRPrimerKeypoints.__init__(
            self,
            src_dict=src_dict,
            dtype=dtype,
            kps=kps,
            mask=mask,
            convention=convention,
            logger=logger)
        if not self.__class__.deprecation_warned:
            self.__class__.deprecation_warned = True
            self.logger.warning(
                'Keypoints defined in XRMoCap is deprecated,' +
                ' use `from xrprimer.data_structure import Keypoints` instead.'
                + ' This class will be removed from XRMoCap before v0.9.0.')
