# Tool start_service

- [Overview](#overview)
- [Argument: config_path](#argument-config_path)
- [Argument: disable_log_file](#argument-disable_log_file)
- [Example](#example)

### Overview

This tool starts a server in the current console according to the configuration file, and sets up a logger. The logger displays information of no less than `INFO` level in the console, and write information of no less than `DEBUG` level in the log file under the `logs/` directory.

For services that use the `work_dir` parameter, please make sure that the target path can be created correctly. Generally speaking, running `mkdir temp` in advance can ensure that the default configuration file in the repository can be successfully used.

### Argument: config_path

`config_path` is the path to a configuration file for server. Please ensure that all parameters required by `SomeService.__init__()` are specified in the configuration file. An example is provided below. For more details, see the docstring in [code](../../../xrmocap/service/base_flask_service.py).

```python
type = 'SMPLStreamService'
name = 'smpl_stream_service'
work_dir = f'temp/{name}'
body_model_dir = 'xrmocap_data/body_models'
device = 'cuda:0'
enable_cors = True
port = 29091
```

Also, you can find our prepared config files in `configs/modules/service/smpl_verts_service.py`.

### Argument: disable_log_file

By default, `disable_log_file` is False and two log files under `logs/f'{service_name}_{time_str}'` will be written. Add `--disable_log_file` makes it True and the tool will only print log to console.

### Example

Run the tool with explicit paths.

```bash
python tools/start_service.py --config_path configs/modules/service/smpl_verts_service.py
```
