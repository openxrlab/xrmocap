# yapf: disable
import gzip
import logging
import numpy as np
import socketio
from enum import Enum
from typing import List, Union

# yapf: enable


class XRMocapSMPLClientActionsEnum(str, Enum):
    UPLOAD = 'upload'
    FORWARD = 'forward'
    GET_FACES = 'get_faces'


class SMPLVertsClient:
    """Client of the XRMocap SMPL Verts server."""

    def __init__(self,
                 server_ip: str = '127.0.0.1',
                 server_port: int = 29091,
                 enable_bytes: bool = True,
                 enable_gzip: bool = False,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Initialize the client.

        Args:
            server_ip (str, optional):
                IP address of the server.
                Defaults to '127.0.0.1'.
            server_port (int, optional):
                Port of the server.
                Defaults to 8376.
            enable_bytes (bool, optional):
                If True, the client will receive bytes from server.
                Otherwise, the client will receive dict.
                Defaults to True.
            enable_gzip (bool, optional):
                If True, the client will decompress the bytes from server.
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        if logger is None or isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger
        self.server_ip = server_ip
        self.server_port = server_port
        self.enable_bytes = enable_bytes
        self.enable_gzip = enable_gzip
        if not self.enable_bytes and self.enable_gzip:
            self.logger.warning('enable_gzip is set to True,' +
                                ' but enable_bytes is set to False. '
                                'enable_gzip will be ignored.')
        # setup websocket client
        self.socketio_client = socketio.Client()
        self.socketio_client.connect(f'http://{server_ip}:{server_port}')

    def _parse_upload_response(self, data):
        if data['status'] == 'success':
            n_frames = int(data['n_frames'])
        else:
            msg = data['msg']
            self.logger.error(
                'Failed to upload body motion, msg from server:\n' + msg)
            self.socketio_client.disconnect()
            raise RuntimeError

        return n_frames

    def upload_smpl_data(self, smpl_data: Union[bytes, str]) -> int:
        """Upload a body motion to the SMPL server.

        Args:
            smpl_data (Union[bytes, str]):
            A SMPL(X)Data file in bytes,
            or a path to SMPL(X)Data file.

        Raises:
            ValueError:
                body_motion is None

        Returns:
            int: number of frames in the body motion
        """
        if isinstance(smpl_data, str):
            with open(smpl_data, 'rb') as f:
                smpl_data_bytes = f.read()
        elif smpl_data is None:
            self.logger.error('SMPL data is None.')
            raise ValueError
        else:
            smpl_data_bytes = smpl_data

        data = {'file_name': 'body_motion', 'file_data': smpl_data_bytes}
        resp_data = self.socketio_client.call(
            XRMocapSMPLClientActionsEnum.UPLOAD, data)
        n_frames = self._parse_upload_response(resp_data)
        return n_frames

    def _parse_get_faces_response(self, data: Union[dict,
                                                    bytes]) -> List[float]:
        # find out if the request is successful first
        if isinstance(data, dict):
            success = (data['status'] == 'success')
        else:
            success = True
        # extract faces according to response type and self settings
        if success:
            if self.enable_bytes:
                bin_data = data
                if self.enable_gzip:
                    bin_data = gzip.decompress(bin_data)
                faces_list = np.frombuffer(
                    bin_data, dtype=np.float16).reshape((-1, 3)).tolist()
            else:
                faces_list = data['faces']
            return faces_list
        else:
            msg = data['msg']
            self.logger.error(msg)
            self.close()
            raise RuntimeError(msg)

    def get_faces(self) -> List[int]:
        """Send a request to get body face indices from the server.

        Returns:
            List[int]: the requested face indices, organized as a [|F|, 3] list
        """
        resp_data = self.socketio_client.call(
            XRMocapSMPLClientActionsEnum.GET_FACES)
        faces = self._parse_get_faces_response(resp_data)
        return faces

    def _parse_forward_response(self, data) -> List[List[float]]:
        # find out if the request is successful first
        if isinstance(data, dict):
            success = (data['status'] == 'success')
        else:
            success = True
        # extract verts according to response type and self settings
        if success:
            if self.enable_bytes:
                bin_data = data
                if self.enable_gzip:
                    bin_data = gzip.decompress(bin_data)
                verts_list = np.frombuffer(
                    bin_data, dtype=np.float16).reshape((-1, 3)).tolist()
            else:
                verts_list = np.asarray(data['verts']).reshape(-1, 3).tolist()
            return verts_list
        else:
            msg = data['msg']
            self.logger.error(msg)
            self.close()
            raise RuntimeError(msg)

    def forward(self, frame_idx: int) -> List[List[float]]:
        """Send a request to get body vertices from the server.

        Args:
            frame_idx (int): frame index in infer

        Returns:
            List[List[float]]:
                A nested list for inferred body vertices,
                shape: [n_verts, 3].
        """
        resp_data = self.socketio_client.call(
            XRMocapSMPLClientActionsEnum.FORWARD, {'frame_idx': frame_idx})
        verts = self._parse_forward_response(resp_data)
        return verts

    def close(self):
        """Close the client."""
        self.socketio_client.disconnect()
