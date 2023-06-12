import os
import shutil
from typing import Tuple, Union
from xrprimer.utils.log_utils import get_logger, logging
from xrprimer.utils.path_utils import Existence, check_path_existence

from xrmocap.utils.service_utils import payload_to_dict

try:
    from flask import Flask, request
    from flask_api import status
    from flask_cors import CORS
    has_flask = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_flask = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class BaseFlaskService:
    """Base http Flask service."""

    def __init__(self,
                 name: str,
                 work_dir: str,
                 debug: bool = False,
                 enable_cors: bool = False,
                 host: str = '0.0.0.0',
                 port: int = 80,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """
        Args:
            name (str): Name of this service.
            work_dir (str):
                Path to a directory, for temp files of this server.
                If empty, no temp files will be created.
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
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)
        # flask also has a global logger: werkzeug
        if not has_flask:
            self.logger.error(import_exception)
            raise ImportError
        self.name = name
        self.debug = debug
        self.host = host
        self.port = port
        self._set_work_dir(work_dir=work_dir)
        self.app = Flask(self.name)
        self.enable_cors = enable_cors
        if self.enable_cors:
            CORS(self.app)
        self.app.add_url_rule(
            '/base_method/', 'base_method', self.base_method, methods=['POST'])

    def _set_work_dir(self, work_dir: str) -> None:
        if len(work_dir) <= 0:
            return
        existence = check_path_existence(work_dir, 'dir')
        if existence == Existence.MissingParent:
            self.logger.error(f'Parent of {work_dir} does not exist.')
            raise FileNotFoundError
        elif existence == Existence.DirectoryExistNotEmpty:
            self.logger.warning('\n' + f'Work dir {work_dir} is not empty!' +
                                ' Please check its content carefully.'
                                ' Clean it and continue? Y/N')
            reply = input().strip().lower()
            if reply == 'y':
                shutil.rmtree(work_dir)
                os.mkdir(work_dir)
            else:
                self.logger.error('Exiting for keeping work_dir safe.')
                raise FileExistsError
        elif existence == Existence.DirectoryNotExist:
            os.mkdir(work_dir)
        self.work_dir = work_dir

    def base_method(self) -> Tuple[dict, int]:
        """A base method for interface testing.

        Returns:
            Tuple[dict, int]:
                dict: Returned payload, or internal error message.
                int: Http response, 200 for OK, 500 for internal error.
        """
        req_json = request.get_json()
        res_dict = payload_to_dict(req_json)
        return res_dict, status.HTTP_200_OK

    def run(self):
        """Run this flask service according to configuration.

        This process will be blocked.
        """
        self.app.run(debug=self.debug, host=self.host, port=self.port)
