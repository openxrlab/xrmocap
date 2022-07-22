# yapf: disable
import logging
import numpy as np
import torch
from typing import List, Union
from xrprimer.utils.log_utils import get_logger

# yapf: enable


class Limbs():
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
        self.logger = get_logger(logger)
        self.connections = None
        self.connection_names = None
        self.parts = None
        self.part_names = None
        self.points = None
        self.logger = get_logger(None)
        self.set_connections(connections, connection_names)
        if parts is not None:
            self.set_parts(parts, part_names)
        if points is not None:
            self.set_points(points)

    def set_connections(self,
                        conn: Union[np.ndarray, torch.Tensor],
                        conn_names: List[str] = None) -> None:
        """Set connection relations of the limbs. Names are optional.

        Args:
            conn (Union[np.ndarray, torch.Tensor]):
                A tensor or ndarray for connections,
                in shape [n_conn, 2],
                conn[:, 0] are start point indice and
                conn[:, 1] are end point indice.
            conn_names (List[str], optional):
                A list of connections names. If given,
                len(conn_names)==len(conn), else default names
                will be returned when getting connections.
                Defaults to None.

        Raises:
            TypeError: Type of connections is not correct.
            ValueError: Shape of connections is not correct.
        """
        if isinstance(conn, torch.Tensor):
            connections = conn.detach().cpu().numpy()
        elif isinstance(conn, np.ndarray):
            connections = conn
        else:
            self.logger.error('Type of connections is not correct.\n' +
                              f'Type: {type(conn)}.')
            raise TypeError
        connections = connections.astype(dtype=np.int32)
        # shape: connection_number, 2
        if connections.shape[-1] != 2 or len(connections.shape) != 2:
            self.logger.error('Shape of connections should be' +
                              ' [n_conn, 2].\n' +
                              f'connections.shape: {connections.shape}.')
            raise ValueError
        self.connections = connections
        if conn_names is not None:
            if len(conn_names) == len(connections):
                self.connection_names = conn_names
            else:
                self.logger.warning(
                    'Length of connection_names is wrong, reset to None.\n' +
                    f'len(conn_names): {len(conn_names)}\n' +
                    f'len(connections): {len(connections)}')
                self.connection_names = None
        else:
            self.connection_names = None

    def set_parts(self,
                  parts: List[List[int]],
                  part_names: List[str] = None) -> None:
        """Set parts of the limbs. If parts has been set, connections can be
        arranged by part when getting. Names are optional.

        Args:
            parts (List[List[int]]):
                A nested list, len(parts) is part number,
                and len(parts[0]) is connection number of the
                first part. Each element in parts[i] is an index
                of one connection.
            part_names (List[str], optional):
                A list of part names. If given,
                len(part_names)==len(parts), else default names
                will be returned when getting parts.
                Defaults to None.

        Raises:
            TypeError: Type of parts is not correct.
            ValueError: Type of connection index is not correct.
        """
        if not isinstance(parts, list):
            self.logger.error('Type of parts is not correct.\n' +
                              f'Type: {type(parts)}.\n' + 'Expect: list')
            raise TypeError
        # Type of conn index: int
        for conn_list in parts:
            for conn_index in conn_list:
                if not isinstance(conn_index, int):
                    self.logger.error(
                        'Type of connection index is not correct.\n' +
                        f'Type: {type(conn_index)}.\n' + 'Expect: int')
                    raise TypeError
        self.parts = parts
        if part_names is not None:
            if len(part_names) == len(parts):
                self.part_names = part_names
            else:
                self.logger.warning(
                    'Length of part_names is wrong, reset to None.\n' +
                    f'len(part_names): {len(part_names)}\n' +
                    f'len(parts): {len(part_names)}')
                self.part_names = None
        else:
            self.part_names = None

    def set_points(self, points: Union[np.ndarray, torch.Tensor]) -> None:
        """Set points of the limbs.

        Args:
            points (Union[np.ndarray, torch.Tensor]):
                A tensor or ndarray for points,
                in shape [n_point, point_dim].

        Raises:
            TypeError:
                Type of points is not correct.
            ValueError:
                Shape of points should be [n_point, point_dim].
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        elif not isinstance(points, np.ndarray):
            self.logger.error('Type of points is not correct.\n' +
                              f'Type: {type(points)}.')
            raise TypeError
        # shape: connection_number, 2
        if len(points.shape) != 2:
            self.logger.error('Shape of points should be' +
                              ' [n_point, point_dim].\n' +
                              f'points.shape: {points.shape}.')
            raise ValueError
        self.points = points

    def clone(self) -> 'Limbs':

        def copy_if_not_None(data):
            if data is None:
                return None
            else:
                return data.copy()

        ret_limbs = Limbs(
            connections=self.connections.copy(),
            connection_names=self.connection_names,
            parts=copy_if_not_None(self.parts),
            part_names=self.part_names,
            points=copy_if_not_None(self.points),
            logger=self.logger)
        return ret_limbs

    def get_points(self) -> Union[np.ndarray, None]:
        """Get points array, which might be None.

        Returns:
            Union[np.ndarray, None]: keypoints
        """
        return self.points

    def get_connections(self) -> np.ndarray:
        """Get connections of limbs.

        Returns:
            np.ndarray: connections
        """
        return self.connections

    def get_connection_names(self) -> List[str]:
        """Get names of connection.

        Returns:
            List[str]: A list of names
        """
        if self.connection_names is None:
            ret_list = [
                f'conn_{conn_index:03d}'
                for conn_index in range(len(self.connections))
            ]
        else:
            ret_list = self.connection_names
        return ret_list

    def get_parts(self) -> Union[List[List[int]], None]:
        """Get parts of limbs, which might be None.

        Returns:
            Union[List[List[int]], None]: parts
        """
        return self.parts

    def get_part_names(self) -> List[str]:
        """Get names of part. If self.part is None, an empty list will be
        returned.

        Returns:
            List[str]: A list of names
        """
        if self.parts is None:
            ret_list = []
        else:
            if self.part_names is None:
                ret_list = [
                    f'part_{part_index:03d}'
                    for part_index in range(len(self.parts))
                ]
            else:
                ret_list = self.part_names
        return ret_list

    def __len__(self) -> int:
        """Get number of connections.

        Returns:
            int
        """
        return len(self.connections)

    def get_connections_in_parts(self) -> dict:
        """Get connection in parts. In each part, there's a list of.

        [start_pt_index, end_pt_index].

        Returns:
            dict: keys are part names.
        """
        ret_dict = {}
        part_names = self.get_part_names()
        for part_index, part_name in enumerate(part_names):
            connection_list = self.parts[part_index]
            ret_dict[part_name] = connection_list
        return ret_dict

    def get_connections_by_names(self) -> dict:
        """Get connection by names.

        Returns:
            dict:
                keys are connection names and
                values are [start_pt_index, end_pt_index],
                in type ndarray.
        """
        ret_dict = {}
        conn_names = self.get_connection_names()
        for conn_index, conn_name in enumerate(conn_names):
            connection = self.connections[conn_index]
            ret_dict[conn_name] = connection
        return ret_dict
