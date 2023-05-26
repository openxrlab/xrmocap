# yapf: disable
import logging
import numpy as np
import torch
from typing import List, Union
from xrprimer.data_structure import Limbs as XRPrimerLimbs

# yapf: enable


class Limbs(XRPrimerLimbs):
    deprecation_warned = False
    """A class for person limbs data, recording connection vectors between
    keypoints.

    Connections are the only necessary data, while human parts, points are
    optional.
    """

    def __init__(self,
                 connections: Union[np.ndarray, torch.Tensor],
                 connection_names: Union[List[str], None] = None,
                 parts: Union[List[List[int]], None] = None,
                 part_names: Union[List[str], None] = None,
                 points: Union[np.ndarray, torch.Tensor, None] = None,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """A class for person limbs data, recording connection vectors between
        keypoints. Connections are the only necessary data. Connections record
        point indice, while parts record connection indice.

        Args:
            connections (Union[np.ndarray, torch.Tensor]):
                A tensor or ndarray for connections,
                in shape [n_conn, 2],
                conn[:, 0] are start point indice and
                conn[:, 1] are end point indice.
            connection_names (Union[List[str], None], optional):
                A list of connections names. If given,
                len(connection_names)==len(conn), else default names
                will be returned when getting connections.
                Defaults to None.
            parts (Union[List[List[int]], None], optional):
                A nested list, len(parts) is part number,
                and len(parts[0]) is connection number of the
                first part. Each element in parts[i] is an index
                of one connection.
            part_names (Union[List[str], None], optional):
                A list of part names. If given,
                len(part_names)==len(parts), else default names
                will be returned when getting parts.
                Defaults to None.
            points (Union[np.ndarray, torch.Tensor, None], optional):
                A tensor or ndarray for points,
                in shape [n_point, point_dim].
                Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        XRPrimerLimbs.__init__(
            self,
            connections=connections,
            connection_names=connection_names,
            parts=parts,
            part_names=part_names,
            points=points,
            logger=logger)
        if not self.__class__.deprecation_warned:
            self.__class__.deprecation_warned = True
            self.logger.warning(
                'Limbs defined in XRMoCap is deprecated,' +
                ' use `from xrprimer.data_structure import Limbs` instead.' +
                ' This class will be removed from XRMoCap before v0.9.0.')
