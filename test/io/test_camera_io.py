from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.io.camera import get_color_camera_parameter_from_smc


def test_load_from_smc():
    smc_reader = SMCReader('test/data/p000103_a000011_tiny.smc')
    kinect_number = smc_reader.num_kinects
    # build triangulator by smc
    for kinect_index in range(kinect_number):
        cam_param = get_color_camera_parameter_from_smc(
            smc_reader=smc_reader,
            camera_type='kinect',
            camera_id=kinect_index)
        break
    assert len(cam_param.get_intrinsic(k_dim=4)) == 4
    # todo: test iphone when it's ready
