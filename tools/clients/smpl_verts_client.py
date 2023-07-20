# yapf: disable
import numpy as np
import socketio
from enum import Enum
from typing import List

# yapf: enable


class XRMocapSMPLClientActionsEnum(str, Enum):
    UPLOAD = 'upload'
    FORWARD = 'forward'
    GET_FACES = 'get_faces'


class XRMocapSMPLClient:
    """Client of the XRMocap SMPL server."""

    def __init__(
        self,
        server_ip: str = '127.0.0.1',
        server_port: int = 8376,
        resp_type: str = 'bytes',
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.resp_type = resp_type

        # setup websocket client
        self.socketio_client = socketio.Client()
        self.socketio_client.connect(f'http://{server_ip}:{server_port}')

        self.upload_success = False
        self.verts = None

    def _parse_upload_response(self, data):
        if data['status'] == 'success':
            num_frames = int(data['num_frames'])
        else:
            msg = data['msg']
            print(f'Failed to upload body motion: {msg}')
            self.socketio_client.disconnect()
            raise RuntimeError

        return num_frames

    def upload_body_motion(self, body_motion: bytes) -> int:
        """Upload a body motion to the SMPL server.

        Args:
            body_motion (bytes): body motion in bytes

        Raises:
            ValueError: raised when the body motion is none

        Returns:
            int: number of frames in the body motion
        """
        if body_motion is None:
            print('Body motion is empty')
            raise ValueError

        data = {'file_name': 'body_motion', 'file_data': body_motion}
        resp_data = self.socketio_client.call(
            XRMocapSMPLClientActionsEnum.UPLOAD, data)
        num_frames = self._parse_upload_response(resp_data)
        print(f'Uploaded body motion, which has {num_frames} frames')
        return num_frames

    def _parse_get_faces_response(self, data):
        success = False
        if self.resp_type == 'bytes':
            if not isinstance(data, dict):
                success = True
                faces = np.frombuffer(
                    data, dtype=np.int32).reshape((-1, 3)).tolist()
            else:
                if data['status'] == 'success':
                    success = True
                    faces = data['faces']

        if not success:
            msg = data['msg']
            print(f'Get faces failed: {msg}')
            self.close()

        return faces

    def get_faces(self) -> List[int]:
        """Send a request to get body face indices from the server.

        Returns:
            List[int]: the requested face indices, organized as a [|F|, 3] list
        """
        resp_data = self.socketio_client.call(
            XRMocapSMPLClientActionsEnum.GET_FACES)
        faces = self._parse_get_faces_response(resp_data)
        print('Got faces')

        return faces

    def _parse_forward_response(self, data):
        success = False
        if self.resp_type == 'bytes':
            if not isinstance(data, dict):
                success = True
                verts = np.frombuffer(data, dtype=np.float16)
        else:
            if data['status'] == 'success':
                success = True
                verts = np.asarray(data['verts'])
        if success:
            verts = verts.reshape(-1, 3)
        else:
            msg = data['msg']
            print(f'Forward failed: {msg}')

        return verts.tolist()

    def forward(self, frame_idx: int) -> np.ndarray:
        """Send a request to get body vertices from the server.

        Args:
            frame_idx (int): frame index in infer

        Returns:
            np.ndarray: the inferred body vertices, organized
                as a [|V|, 3] ndarray
        """
        resp_data = self.socketio_client.call(
            XRMocapSMPLClientActionsEnum.FORWARD, {'frame_idx': frame_idx})
        verts = self._parse_forward_response(resp_data)

        return verts

    def close(self):
        """Close the client."""
        self.socketio_client.disconnect()
