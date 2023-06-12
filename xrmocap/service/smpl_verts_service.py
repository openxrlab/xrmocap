# yapf: disable
import numpy as np
import os
import time
import torch
from flask_socketio import SocketIO, emit
from threading import RLock
from typing import Union
from xrprimer.utils.log_utils import logging

from xrmocap.data_structure.body_model import auto_load_smpl_data
from xrmocap.model.body_model.builder import build_body_model
from xrmocap.utils.time_utils import Timer
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


class SMPLVertsService(BaseFlaskService):
    """TODO: Docstring."""

    def __init__(self,
                 name: str,
                 body_model_dir: str,
                 work_dir: str,
                 flat_hand_mean: bool = False,
                 debug: bool = False,
                 enable_cors: bool = False,
                 device: Union[torch.device, str] = 'cuda',
                 host: str = '0.0.0.0',
                 port: int = 29091,
                 logger: Union[None, str, logging.Logger] = None) -> None:
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
        self.socketio = SocketIO(self.app)
        self.device = device
        self.worker_lock = RLock()
        # set body model configs for all types and genders
        # stored in self.body_model_configs
        self._set_body_model_config(body_model_dir, flat_hand_mean)
        # save clients' uuid, body model type, gender and
        # last connection time in self.client_config_cache
        self.client_config_cache = {}
        self.socketio.on_event(
            message='upload',
            handler=self.upload_smpl_data,
        )
        self.socketio.on_event(
            message='forward',
            handler=self.forward_body_model,
        )
        self.socketio.on_event(
            message='disconnect',
            handler=self.on_disconnect,
        )
        self.forward_timer = Timer(
            name='forward_timer',
            logger=self.logger,
        )

    def run(self):
        """Run this flask service according to configuration.

        This process will be blocked.
        """
        self.socketio.run(
            app=self.app, debug=self.debug, host=self.host, port=self.port)

    def upload_smpl_data(self, data: dict) -> dict:
        """Upload smpl data file, check whether the corresponding body model
        config exists, and save it to work_dir if success.

        Args:
            data (dict): smpl data file info, including
                uuid, file_name and file_data.

        Returns:
            dict: response info, including status, and
                msg when fails.
        """
        resp_dict = dict()
        # save file to work_dir
        uuid = data['uuid']
        if uuid in self.client_config_cache:
            warn_msg = f'Client {uuid} already registered.' +\
                ' Overwrite its body model config.'
            self.logger.warning(warn_msg)
            resp_dict['msg'] = f'Warning: {warn_msg}'
        file_name = data['file_name']
        file_data = data['file_data']
        file_path = os.path.join(self.work_dir, f'{uuid}_{file_name}.npz')
        with open(file_path, 'wb') as file:
            file.write(file_data)
        # load smpl data
        smpl_data, class_name = auto_load_smpl_data(
            npz_path=file_path, logger=self.logger)
        smpl_type = class_name.replace('Data', '').lower()
        smpl_gender = smpl_data.get_gender()
        # check if the body model files exist
        if smpl_type not in self.body_model_configs or\
                smpl_gender not in self.body_model_configs[smpl_type]:
            error_msg = f'Client {uuid} has smpl type {smpl_type} ' +\
                f'and smpl gender {smpl_gender}, ' +\
                'but no corresponding body model config found.'
            resp_dict['msg'] = f'Error: {warn_msg}'
            self.logger.error(error_msg)
            emit('upload_response', resp_dict)
        # build body model
        body_model_cfg = self.body_model_configs[smpl_type][smpl_gender]
        body_model = build_body_model(body_model_cfg).to(self.device)
        # save body model to cache
        self.client_config_cache[uuid] = dict(
            smpl_type=smpl_type.replace('Data', '').lower(),
            smpl_gender=smpl_data.get_gender(),
            smpl_data=smpl_data,
            body_model=body_model,
            last_connect_time=time.time())
        self.logger.info(f'Client {uuid} smpl data file loaded confirmed.\n' +
                         f'Body model type: {smpl_type}\n' +
                         f'Gender: {smpl_gender}')
        resp_dict['status'] = 'success'
        emit('upload_response', resp_dict)
        return resp_dict

    def forward_body_model(self, data: dict) -> dict:
        """Call body_model.forward() to get SMPL vertices.

        Args:
            data (dict): Request data. uuid, frame_idx are required.

        Returns:
            dict: Response data.
                If success, status is 'success' and
                vertices is a list of vertices.
        """
        resp_dict = dict()
        req_dict = data
        uuid = req_dict['uuid']
        frame_idx = req_dict['frame_idx']
        smpl_data = self.client_config_cache[uuid]['smpl_data']
        # check if data and args are valid
        failed = False
        if uuid not in self.client_config_cache:
            error_msg = f'Client {uuid} not registered.'
            failed = True
        elif smpl_data is None:
            error_msg = f'Client {uuid}\'s smpl data not uploaded.'
            failed = True
        elif frame_idx >= smpl_data.get_batch_size():
            error_msg = f'Client {uuid}\'s smpl data only has ' +\
                f'{smpl_data.get_batch_size()} frames, ' +\
                f'but got frame_idx {frame_idx} in request.'
            failed = True
        if failed:
            self.logger.error(error_msg)
            resp_dict['msg'] = f'Error: {error_msg}'
            resp_dict['status'] = 'fail'
            emit('forward_response', resp_dict)
        else:
            self.forward_timer.start()
            tensor_dict = smpl_data.to_tensor_dict(
                repeat_betas=True, device=self.device)
            for k, v in tensor_dict.items():
                tensor_dict[k] = v[frame_idx:frame_idx + 1]
            body_model = self.client_config_cache[uuid]['body_model']
            with self.worker_lock:
                with torch.no_grad():
                    body_model_output = body_model(**tensor_dict)
            verts = body_model_output['vertices']  # n_batch=1, n_verts, 3
            verts_np = verts.cpu().numpy().squeeze(0).astype(np.float16)
            self.client_config_cache[uuid]['last_connect_time'] = time.time()
            resp_dict['verts'] = verts_np.tolist()
            resp_dict['status'] = 'success'
            self.forward_timer.stop()
            if self.forward_timer.count >= 50:
                self.logger.info(
                    'Average forward time per-frame:' +
                    f' {self.forward_timer.get_average(reset=True):.4f} s')
            emit('forward_response', resp_dict)
        return resp_dict

    def on_disconnect(self, data: dict) -> None:
        """Disconnect event handler.

        Args:
            data (dict): Request data. uuid is required.
        """
        req_dict = data
        uuid = req_dict['uuid']
        self.logger.info(
            f'Client {uuid} disconnected. Cleaning files and cache.')
        self._clean_by_uuid(uuid)

    def _clean_by_uuid(self, uuid: str) -> None:
        file_names = os.listdir(self.work_dir)
        for file_name in file_names:
            if file_name.startswith(uuid):
                file_path = os.path.join(self.work_dir, file_name)
                os.remove(file_path)
        self.client_config_cache.pop(uuid, None)

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
