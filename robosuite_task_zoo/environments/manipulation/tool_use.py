from collections import OrderedDict
import numpy as np
from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material
from robosuite.utils.buffers import RingBuffer
import robosuite.utils.transform_utils as T

from robosuite_task_zoo.models.tool_use import LShapeTool, PotObject

class ToolUseEnvBase(SingleArmEnv):
    """
    Kitchen Env: The task is: place plate on the stove, cook with different ingradients and place the plate on the serving region.
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=False,
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
        contact_threshold=2.0,
        cube_x_range = [0.29, 0.30],
        cube_y_range = [-0.14, -0.12],
        cube_rotation_range = (-np.pi / 2, -np.pi / 2),
        tool_x_range = [0.07, 0.07],
        tool_y_range = [-0.05, -0.05],
        tool_rotation_range = (0., 0.),
            
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_offset = (-0.2, 0, 0.90)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # Thresholds
        self.contact_threshold = contact_threshold

        # History observations
        self._history_force_torque = None
        self._recent_force_torque = None
        
        self.objects = []

        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range
        self.cube_rotation_range = cube_rotation_range

        self.tool_x_range = tool_x_range
        self.tool_y_range = tool_y_range
        self.tool_rotation_range = tool_rotation_range
        

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

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the drawer is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between drawer handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by drawer handled
              - Note that this component is only relevant if the environment is using the locked drawer version

        Note that a successfully completed task (drawer opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

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
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414,
            0.49641484022140503, 0.6968992352485657]
        )
        
        
        # initialize objects of interest
        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ingredient_size = [0.02, 0.025, 0.02]
        self.cube = BoxObject(
            name="cube_bread",
            size_min=ingredient_size,
            size_max=ingredient_size,
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
        
        # Create placement initializer
        self.placement_initializer.append_sampler(
        sampler = UniformRandomSampler(
            name="ObjectSampler-cube",
            mujoco_objects=self.cube,
            x_range=self.cube_x_range,
            y_range=self.cube_y_range,
            rotation=self.cube_rotation_range,
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
            x_range=self.tool_x_range,
            y_range=self.tool_y_range,
            rotation=self.tool_rotation_range,
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
        
        mujoco_objects = [
            self.pot_object,
            self.cube,
            self.lshape_tool
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
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
        self.object_body_ids = dict()

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

        observables["robot0_joint_pos"]._active = True
        
        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            sensors = []
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked drawer version with rotatable handle

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)

        sensors.append(world_pose_in_gripper)
        names.append("world_pose_in_gripper")

        @sensor(modality=modality)
        def gripper_contact(obs_cache):
            return self._has_gripper_contact

        @sensor(modality=modality)
        def force_norm(obs_cache):
            return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)

        sensors += [gripper_contact, force_norm]
        names += [f"{pf}contact", f"{pf}eef_force_norm"]

        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq
                )
                
        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names
    
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

    def _check_success(self):
        """
        Check if drawer has been opened.

        Returns:
            bool: True if drawer has been opened
        """

        pot_pos = self.sim.data.body_xpos[self.pot_object_id]
        cube_pos = self.sim.data.body_xpos[self.cube_id]
        object_in_pot = self.check_contact(self.cube, self.pot_object) and np.linalg.norm(pot_pos[:2] - cube_pos[:2]) < 0.06 and np.abs(pot_pos[2] - cube_pos[2]) < 0.05
        
        return object_in_pot

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4:
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)
        
        self._recent_force_torque = []
        obs, reward, done, info = super().step(action)
        info["history_ft"] = np.clip(np.copy(self._history_force_torque.buf), a_min=None, a_max=2)
        info["recent_ft"] = np.array(self._recent_force_torque)
        done = self._check_success()
        return obs, reward, done, info
        
        
    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

        self._history_force_torque.push(np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))
        self._recent_force_torque.append(np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))
        
    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque
            
        return reward, done, info
        
    @property
    def _has_gripper_contact(self):
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold

    
    def get_state_vector(self, obs):
        return np.concatenate([obs["robot0_gripper_qpos"],
                               obs["robot0_eef_pos"],
                               obs["robot0_eef_quat"]])

    def _post_process(self):
        pass

class ToolUseEnv(ToolUseEnvBase):
    """Hardest varaint"""
    def __init__(self, *args, **kwargs):
        kwargs["cube_x_range"] = [0.29, 0.32]
        kwargs["cube_y_range"] = [-0.25, -0.10]
        kwargs["tool_x_range"] = [0.06, 0.08]
        kwargs["tool_y_range"] = [-0.23, -0.03]
        
        super().__init__(*args, **kwargs)

