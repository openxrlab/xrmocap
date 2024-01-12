# yapf: disable
import gzip
import numpy as np
import os
import time
import torch
import uuid
from flask import session
from flask_socketio import SocketIO, emit
from threading import RLock
from typing import Union
from xrprimer.utils.log_utils import logging

from xrmocap.data_structure.body_model import auto_load_smpl_data
from xrmocap.model.body_model.builder import build_body_model
from xrmocap.utils.time_utils import Timer
from xrmocap.utils.data_convert_utils import SMPLDataConverter, SMPLDataTypeEnum
from .base_flask_service import BaseFlaskService

# yapf: enable

_SMPL_CONFIG_TEMPLATE = dict(
    type='SMPL',
    gender='neutral',
    num_betas=10,
    keypoint_convention='smpl_45',
    model_path='xrmocap_data/body_models/smpl',
    batch_size=1)
_SMPLX_CONFIG_TEMPLATE = dict(
    type='SMPLX',
    gender='neutral',
    num_betas=10,
    keypoint_convention='smplx',
    model_path='xrmocap_data/body_models/smplx',
    batch_size=1,
    use_face_contour=True,
    use_pca=False,
    flat_hand_mean=False)


class SMPLStreamService(BaseFlaskService):
    """A websocket service that provides SMPL/SMPLX vertices in stream."""

    def __init__(self,
                 name: str,
                 body_model_dir: str,
                 work_dir: str,
                 secret_key: Union[None, str] = None,
                 flat_hand_mean: bool = False,
                 enable_bytes: bool = True,
                 enable_gzip: bool = False,
                 debug: bool = False,
                 enable_cors: bool = False,
                 device: Union[torch.device, str] = 'cuda',
                 host: str = '0.0.0.0',
                 port: int = 29091,
                 max_http_buffer_size: int = 128 * 1024 * 1024,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """
        Args:
            name (str): Name of this service.
            body_model_dir (str):
                Path to the directory for SMPL(X) body models, folder `smpl`
                and `smplx` are below body_model_dir.
            work_dir (str):
                Path to a directory, for temp files of this server.
            secret_key (Union[None, str], optional):
                Secret key for this service. If None, a random key will be
                generated. Defaults to None.
            flat_hand_mean (bool, optional):
                If False, then the pose of the hand is initialized to False.
                Defaults to False.
            enable_bytes (bool, optional):
                Whether to enable bytes response. Defaults to True.
            enable_gzip (bool, optional):
                Whether to enable gzip compression for the verts response.
                Defaults to False.
            debug (bool, optional):
                If `debug` flag is set the server will automatically reload
                for code changes and show a debugger in case
                an exception happened.
                Defaults to False.
            enable_cors (bool, optional):
                Whether to enable Cross Origin Resource Sharing (CORS).
                Defaults to False.
            host (str, optional):
                Host IP address. 127.0.0.1 for localhost,
                0.0.0.0 for all local network interfaces.
                Defaults to '0.0.0.0'.
            port (int, optional):
                Port for this http service.
                Defaults to 80.
            max_http_buffer_size (int):
                Server's payload.
                Defaults to 128MB.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseFlaskService.__init__(
            self,
            name=name,
            work_dir=work_dir,
            debug=debug,
            enable_cors=enable_cors,
            host=host,
            port=port,
            logger=logger,
        )
        self.app.config['SECRET_KEY'] = os.urandom(24) \
            if secret_key is None \
            else secret_key
        # max_http_buffer_size: the maximum allowed payload
        self.socketio = SocketIO(
            self.app, max_http_buffer_size=max_http_buffer_size)
        self.device = device
        self.worker_lock = RLock()
        # set body model configs for all types and genders
        # stored in self.body_model_configs
        self._set_body_model_config(body_model_dir, flat_hand_mean)
        # set enable_bytes and enable_gzip
        self.enable_bytes = enable_bytes
        self.enable_gzip = enable_gzip
        if not self.enable_bytes and self.enable_gzip:
            self.logger.warning('enable_gzip is set to True,' +
                                ' but enable_bytes is set to False. '
                                'enable_gzip will be ignored.')

        self.socketio.on_event('upload', self.upload_smpl_data)

        self.socketio.on_event('forward', self.forward_body_model)

        self.socketio.on_event('get_faces', self.get_faces)

        self.socketio.on_event(
            message='disconnect',
            handler=self.on_disconnect,
        )
        self.socketio.on_event(
            message='connect',
            handler=self.on_connect,
        )
        self.forward_timer = Timer(
            name='forward_timer',
            logger=self.logger,
        )

        self.data_converter = SMPLDataConverter(logger=self.logger)

    def run(self):
        """Run this flask service according to configuration.

        This process will be blocked.
        """
        self.socketio.run(
            app=self.app, debug=self.debug, host=self.host, port=self.port)

    def on_connect(self) -> None:
        """Connect event handler.

        Register client uuid.
        """
        uuid_str = str(uuid.uuid4())
        session['uuid'] = uuid_str
        self.logger.info(f'Client {uuid_str} connected.')

    def on_disconnect(self) -> None:
        """Disconnect event handler.

        Args:
            data (dict): Request data. uuid is required.
        """
        uuid_str = session['uuid']
        self.logger.info(
            f'Client {uuid_str} disconnected. Cleaning files and session.')
        self._clean_files_by_uuid(uuid_str)
        session.clear()

    def upload_smpl_data(self, data: dict) -> dict:
        """Upload smpl data file, check whether the corresponding body model
        config exists, and save it to work_dir if success.

        Args:
            data (dict): smpl data file info, including
                file_name and file_data.

        Returns:
            dict: response info, including status, and
                msg when fails.
        """

        resp_dict = dict()
        uuid_str = session['uuid']
        smpl_data_in_session = session.get('smpl_data', None)
        if smpl_data_in_session is not None:
            warn_msg = f'Client {uuid_str} has already uploaded a file.' +\
                ' Overwriting.'
            self.logger.warning(warn_msg)
            resp_dict['msg'] = f'Warning: {warn_msg}'
        file_name = data['file_name']
        file_data = data['file_data']
        file_path = os.path.join(self.work_dir, f'{uuid_str}_{file_name}.npz')
        with open(file_path, 'wb') as file:
            file.write(file_data)
        data_type = self.data_converter.get_data_type(file_path)
        # organize the input data as the smpl data
        if data_type is SMPLDataTypeEnum.AMASS:
            self.logger.info('Received AMASS data, converting to SMPL(X) data')
            data = self.data_converter.from_amass(file_path)
            data.dump(file_path)
        elif data_type is SMPLDataTypeEnum.HUMANDATA:
            self.logger.info('Received HumanData, converting to SMPL(X) data')
            data = self.data_converter.from_humandata(file_path)
            data.dump(file_path)
        elif data_type is SMPLDataTypeEnum.UNKNOWN:
            error_msg = 'Failed to convert uploaded data due to ' + \
                'unknown data type, supported data types: ' + \
                f'{[e.value for e in SMPLDataTypeEnum if e is not SMPLDataTypeEnum.UNKNOWN]}'
            self.logger.error(error_msg)
            resp_dict['msg'] = f'Error: {error_msg}'
            resp_dict['status'] = 'fail'
            return resp_dict
        # load smpl data
        smpl_data, class_name = auto_load_smpl_data(
            npz_path=file_path, logger=self.logger)
        smpl_type = class_name.replace('Data', '').lower()
        smpl_gender = smpl_data.get_gender()
        # check if the body model files exist
        if smpl_type not in self.body_model_configs or\
                smpl_gender not in self.body_model_configs[smpl_type]:
            error_msg = f'Client {uuid_str} has smpl type {smpl_type} ' +\
                f'and smpl gender {smpl_gender}, ' +\
                'but no corresponding body model config found.'
            resp_dict['msg'] = f'Error: {error_msg}'
            self.logger.error(error_msg)
            return resp_dict
        # build body model
        body_model_cfg = self.body_model_configs[smpl_type][smpl_gender]
        body_model = build_body_model(body_model_cfg).to(self.device)
        # save body model to cache
        session['smpl_type'] = smpl_type.replace('Data', '').lower()
        session['smpl_gender'] = smpl_data.get_gender()
        session['smpl_data'] = smpl_data
        session['body_model'] = body_model
        session['last_connect_time'] = time.time()
        self.logger.info(
            f'Client {uuid_str} smpl data file loaded confirmed.\n' +
            f'Body model type: {smpl_type}\n' + f'Gender: {smpl_gender}')
        resp_dict['n_frames'] = smpl_data.get_batch_size()
        resp_dict['status'] = 'success'

        return resp_dict

    def forward_body_model(self, data: dict) -> dict:
        """Call body_model.forward() to get SMPL vertices.

        Args:
            data (dict): Request data, frame_idx is required.

        Returns:
            dict: Response data.
                If success, status is 'success' and
                vertices bytes for an ndarray.
        """
        resp_dict = dict()
        req_dict = data
        uuid_str = session['uuid']
        frame_idx = req_dict['frame_idx']
        smpl_data = session['smpl_data']
        # check if data and args are valid
        failed = False
        if smpl_data is None:
            error_msg = f'Client {uuid_str}\'s smpl data not uploaded.'
            failed = True
        elif frame_idx >= smpl_data.get_batch_size():
            error_msg = f'Client {uuid_str}\'s smpl data only has ' +\
                f'{smpl_data.get_batch_size()} frames, ' +\
                f'but got frame_idx={frame_idx} in request.'
            failed = True
        if failed:
            self.logger.error(error_msg)
            resp_dict['msg'] = f'Error: {error_msg}'
            resp_dict['status'] = 'fail'
            return resp_dict
        # no error, forward body model
        else:
            tensor_dict = smpl_data.to_tensor_dict(
                repeat_betas=True, device=self.device)
            for k, v in tensor_dict.items():
                tensor_dict[k] = v[frame_idx:frame_idx + 1]
            body_model = session['body_model']
            with self.worker_lock:
                self.forward_timer.start()
                with torch.no_grad():
                    body_model_output = body_model(**tensor_dict)
                self.forward_timer.stop()
                if self.forward_timer.count >= 50:
                    self.logger.info(
                        'Average forward time per-frame:' +
                        f' {self.forward_timer.get_average(reset=True):.4f} s')
            verts = body_model_output['vertices']  # n_batch=1, n_verts, 3
            verts_np = verts.cpu().numpy().squeeze(0).astype(np.float16)
            session['last_connect_time'] = time.time()
            if self.enable_bytes:
                verts_bytes = verts_np.tobytes()
                if self.enable_gzip:
                    verts_bytes = gzip.compress(verts_bytes)
                return verts_bytes
            else:
                resp_dict['verts'] = verts_np.tolist()
                resp_dict['status'] = 'success'
                return resp_dict

    def get_faces(self) -> dict:
        """Get body face indices.

        Returns:
            dict: Response data.
                If success, status is 'success' and
                face indices for an ndarray.
        """
        resp_dict = dict()
        body_model = session['body_model']
        # check if data and args are valid
        if body_model is None:
            error_msg = 'Failed to get body model.'
            self.logger.error(error_msg)
            resp_dict['msg'] = f'Error: {error_msg}'
            resp_dict['status'] = 'fail'

            return resp_dict

        session['last_connect_time'] = time.time()

        if self.enable_bytes:
            faces = np.array(body_model.faces, dtype=np.int32)
            faces_bytes = faces.tobytes()
            if self.enable_gzip:
                faces_bytes = gzip.compress(faces_bytes)
            return faces_bytes
        else:
            resp_dict['faces'] = faces
            resp_dict['status'] = 'success'
            return resp_dict

    def _clean_files_by_uuid(self, uuid: str) -> None:
        file_names = os.listdir(self.work_dir)
        for file_name in file_names:
            if file_name.startswith(uuid):
                file_path = os.path.join(self.work_dir, file_name)
                os.remove(file_path)

    def _set_body_model_config(self, body_model_dir: str,
                               flat_hand_mean: bool) -> None:
        self.body_model_dir = body_model_dir
        self.flat_hand_mean = flat_hand_mean
        genders = ('neutral', 'female', 'male')
        smpl_configs = dict()
        smplx_configs = dict()
        absent_models = []
        for gender in genders:
            file_name = f'SMPL_{gender.upper()}.pkl'
            file_path = os.path.join(body_model_dir, 'smpl', file_name)
            if os.path.exists(file_path):
                gender_config = _SMPL_CONFIG_TEMPLATE.copy()
                gender_config['gender'] = gender
                gender_config['logger'] = self.logger
                gender_config['model_path'] = os.path.join(
                    body_model_dir, 'smpl')
                smpl_configs[gender] = gender_config
            else:
                absent_models.append(file_name)
            file_name = f'SMPLX_{gender.upper()}.npz'
            file_path = os.path.join(body_model_dir, 'smplx', file_name)
            if os.path.exists(file_path):
                gender_config = _SMPLX_CONFIG_TEMPLATE.copy()
                gender_config['gender'] = gender
                gender_config['logger'] = self.logger
                gender_config['flat_hand_mean'] = flat_hand_mean
                gender_config['model_path'] = os.path.join(
                    body_model_dir, 'smplx')
                smplx_configs[gender] = gender_config
            else:
                absent_models.append(file_name)
        self.body_model_configs = dict(
            smpl=smpl_configs,
            smplx=smplx_configs,
        )
        if len(smpl_configs) + len(smplx_configs) <= 0:
            self.logger.error(f'No body_model found below {body_model_dir}.')
            raise FileNotFoundError
        if len(absent_models) > 0:
            self.logger.warning(f'Missing {len(absent_models)} model files.' +
                                ' The following models cannot be used:\n' +
                                f'{absent_models}')
