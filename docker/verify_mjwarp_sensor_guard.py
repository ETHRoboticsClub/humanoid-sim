"""GPU differential check: skip unused tactile work, retain other sensors/RNE."""
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from apply_mjwarp_patch import SENSOR_AFTER, SENSOR_BEFORE
from mujoco_warp._src import sensor


def verify(reference):
    for enabled in (False, True, 'tactile'):
        sensors = '<sensor><accelerometer site="s"/><gyro site="s"/><touch site="s"/><force site="s"/><torque site="s"/></sensor>' if enabled else ''
        xml = '''<mujoco><option timestep=".001"/><worldbody><geom type="plane" size="1 1 .1"/>
          <body pos="0 0 .095"><freejoint/><geom type="sphere" size=".1" mass="1"/>
          <site name="s" type="sphere" size=".11"/></body></worldbody>''' + sensors + '</mujoco>'
        if enabled == 'tactile':
            xml = '''<mujoco><option><flag multiccd="enable"/></option>
              <asset><mesh name="mesh" builtin="sphere" params="0"/></asset>
              <worldbody><body pos="0 0 1"><freejoint/><geom name="skin" type="mesh" mesh="mesh"/></body>
              <body><geom type="box" size=".7 .7 .3"/></body></worldbody>
              <sensor><tactile geom="skin" mesh="mesh"/></sensor></mujoco>'''
        cpu_model = mujoco.MjModel.from_xml_string(xml)  # parsing only; no CPU physics
        model = mjw.put_model(cpu_model)
        data = mjw.make_data(cpu_model, nworld=16, nconmax=8, njmax=32)
        model.sensor_rne_postconstraint = True  # Newton requests this for body acceleration.
        assert (model.nsensortaxel > 0) == (enabled == 'tactile')
        mjw.forward(model, data)
        assert data.nacon.numpy().sum() > 0, "fixture must exercise contacts"
        fields = ('sensordata', 'cacc', 'cfrc_int', 'cfrc_ext')
        for name in fields:
            getattr(data, name).zero_()
        reference.sensor_acc(model, data)
        expected = {name: getattr(data, name).numpy().copy() for name in fields}
        assert np.any(expected['cacc']), "fixture must exercise body acceleration"
        if enabled:
            assert np.any(expected['sensordata']), "fixture must produce sensor output"
        for name in fields:
            assert np.isfinite(expected[name]).all()
            getattr(data, name).zero_()
        sensor.sensor_acc(model, data)
        for name in fields:
            np.testing.assert_array_equal(expected[name], getattr(data, name).numpy(), err_msg=f'{enabled=} {name=}')
        print('SENSOR_GUARD_PASS', enabled, flush=True)


if __name__ == "__main__":
    wp.init()
    if not wp.get_cuda_devices():
        raise RuntimeError("Sensor differential verification requires CUDA")
    wp.set_device("cuda:0")
    source = Path(sensor.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SENSOR_AFTER
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "_src").mkdir()
        pristine = root / "_src/sensor.py"
        shutil.copy2(source, pristine)
        patch = Path(__file__).parent / "patches/mujoco-warp-3.12-no-unused-tactile.patch"
        subprocess.run(["git", "apply", "--reverse", str(patch.resolve())], cwd=root, check=True)
        assert hashlib.sha256(pristine.read_bytes()).hexdigest() == SENSOR_BEFORE
        spec = importlib.util.spec_from_file_location("mjwarp_sensor_reference", pristine)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        verify(reference)
