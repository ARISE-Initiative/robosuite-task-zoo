import robosuite as suite
from robosuite.devices import SpaceMouse
from robosuite.utils.input_utils import input2action

import robosuite_task_zoo

import robosuite.utils.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from robosuite.environments.base import REGISTERED_ENVS, MujocoEnv

# device = SpaceMouse(9583, 50734, pos_sensitivity=1.0, rot_sensitivity=0.1)
# device.start_control()

options = {}
options["robots"] = ["Panda"]

options["controller_configs"] = suite.load_controller_config(default_controller="OSC_POSITION")

options["env_name"] = "MultitaskKitchenDomain"


env = suite.make(**options,
                 has_renderer=True,
                 has_offscreen_renderer=False,
                 ignore_done=True,
                 use_camera_obs=False,
                 horizon=100,
                 control_freq=20,
                 task_id=0)


env.reset()
for _ in range(100):
    env.step([1.] * 4)    
    env.render()
