onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=[288, 384],
    optimize=True)
codebase_config = dict(type='mmpose', task='PoseDetection')
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=False, max_workspace_size=1073741824),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 384, 288],
                    opt_shape=[1, 3, 384, 288],
                    max_shape=[1, 3, 384, 288])))
    ])
