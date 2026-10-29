"""Compare complete GPU steps, including diagnostic outputs and callback observations."""
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
from apply_mjwarp_patch import FORWARD_AFTER, FORWARD_BEFORE
from mujoco_warp._src import forward


def verify(reference):
    for integrator, observed in [('RK4', 'none'), ('RK4', 'sensor'), ('RK4', 'callback'), ('Euler', 'none'), ('implicitfast', 'none')]:
        sensors = '<sensor><accelerometer site="s"/></sensor>' if observed == 'sensor' else ''
        xml = f'''<mujoco><option timestep=".001" integrator="{integrator}"/><worldbody>
          <geom type="plane" size="1 1 .1"/><body pos="0 0 .095"><freejoint/>
          <geom type="sphere" size=".1" mass="1"/><site name="s"/></body></worldbody>{sensors}</mujoco>'''
        cpu_model = mujoco.MjModel.from_xml_string(xml)
        model = mjw.put_model(cpu_model)
        model.sensor_rne_postconstraint = True
        old = mjw.make_data(cpu_model, nworld=16, nconmax=8, njmax=32)
        new = mjw.make_data(cpu_model, nworld=16, nconmax=8, njmax=32)
        velocity = wp.array(np.random.default_rng(41).normal(0, .1, (16, cpu_model.nv)).astype(np.float32))
        wp.copy(old.qvel, velocity)
        wp.copy(new.qvel, velocity)
        callbacks = []
        if observed == 'callback':
            model.callback.control = lambda m, d: callbacks.append(d.cacc.numpy().copy())
        fields = ('qpos', 'qvel', 'qacc', 'qfrc_constraint', 'cacc', 'cfrc_int', 'cfrc_ext', 'sensordata')
        for step in range(5):
            callbacks.clear()
            reference.step(model, old)
            old_callbacks = callbacks.copy()
            callbacks.clear()
            forward.step(model, new)
            assert len(old_callbacks) == len(callbacks)
            for x, y in zip(old_callbacks, callbacks):
                np.testing.assert_array_equal(x, y)
            for name in fields:
                np.testing.assert_array_equal(getattr(old, name).numpy(), getattr(new, name).numpy(),
                                              err_msg=f'{integrator=} {observed=} {step=} {name=}')
        print('RK4_STAGES_PASS', integrator, observed, flush=True)


if __name__ == "__main__":
    wp.init()
    if not wp.get_cuda_devices():
        raise RuntimeError("RK4 differential verification requires CUDA")
    wp.set_device("cuda:0")
    source = Path(forward.__file__)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == FORWARD_AFTER
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "_src").mkdir()
        pristine = root / "_src/forward.py"
        shutil.copy2(source, pristine)
        patch = Path(__file__).parent / "patches/mujoco-warp-3.12-rk4-sensor-stages.patch"
        subprocess.run(["git", "apply", "--reverse", str(patch.resolve())], cwd=root, check=True)
        assert hashlib.sha256(pristine.read_bytes()).hexdigest() == FORWARD_BEFORE
        spec = importlib.util.spec_from_file_location("mjwarp_forward_reference", pristine)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        verify(reference)
