from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor

from robosuite_task_zoo.models.tool_use import LShapeTool, PotObject


class ToolUse(SingleArmEnv):
    """
    This class corresponds to the tool use task for a single robot arm.

    Args:
        TODO

    Raises:
        TODO
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.cube = BoxObject(
            name="cube",
            size_min=[0.02, 0.025, 0.02],
            size_max=[0.02, 0.025, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
            density=500.,
        )

        self.lshape_tool = LShapeTool(
            name="LShapeTool",
        )
        self.pot_object = PotObject(
            name="PotObject",
        )
        pot_object = self.pot_object.get_obj(); pot_object.set("pos", array_to_string((0.0, 0.18, self.table_offset[2] + 0.05)))
        
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
        sampler = UniformRandomSampler(
            name="ObjectSampler-cube",
            mujoco_objects=self.cube,
            x_range=[0.29, 0.32],
            y_range=[-0.25, -0.10],
            rotation=(-np.pi / 2., -np.pi / 2.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler = UniformRandomSampler(
            name="ObjectSampler-lshape",
            mujoco_objects=self.lshape_tool,
            x_range=[0.06,  0.08],
            y_range=[-0.23, -0.03],
            rotation=(0., 0.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
            
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[self.cube,
                            self.lshape_tool,
                            self.pot_object],
        )

        self.objects = [
            self.pot_object,
            self.cube,
            self.lshape_tool
        ]

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.pot_object_id = self.sim.model.body_name2id(self.pot_object.root_body)
        self.lshape_tool_id = self.sim.model.body_name2id(self.lshape_tool.root_body)
        self.cube_id = self.sim.model.body_name2id(self.cube.root_body)

        self.obj_body_id = {}        
        for obj in self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
        

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been placed in the pot.

        Returns:
            bool: True if cube has been lifted
        """
        pot_pos = self.sim.data.body_xpos[self.pot_object_id]
        cube_pos = self.sim.data.body_xpos[self.cube_id]
        object_in_pot = self.check_contact(self.cube, self.pot_object) and np.linalg.norm(pot_pos[:2] - cube_pos[:2]) < 0.06 and np.abs(pot_pos[2] - cube_pos[2]) < 0.05
        
        return object_in_pot


    def step(self, action):
        obs, reward, done, info = super().step(action)
        done = self._check_success()
        return obs, reward, done, info

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)


    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        return reward, done, info
