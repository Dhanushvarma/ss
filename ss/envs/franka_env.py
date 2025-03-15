import numpy as np
import time
import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import mediapy as media


class FrankaEnv(gym.Env):
    """Gym environment for Franka Panda arm with operational space control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None, xml_path=None):
        """Initialize environment with operational space controller.

        Args:
            render_mode: The render mode to use, either "human" or "rgb_array"
            xml_path: Path to the MuJoCo XML file for the Franka scene
        """
        # Default XML path
        if xml_path is None:
            pwd = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(pwd, "../../assets/franka_emika_panda/scene.xml")

        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Simulation settings
        self.dt = 0.001
        self.model.opt.timestep = self.dt

        # Controller constants
        self.home_joint_pos = [
            0,
            0,
            0,
            -1.57079,
            0,
            1.57079,
            -2.29,
        ]  # except the hand joints
        self.impedance_pos = np.array([500.0, 500.0, 500.0])  # [N/m]
        self.impedance_ori = np.array([250.0, 250.0, 250.0])  # [Nm/rad]
        self.Kp_null = np.array([10.0] * 7)  # NOTE: got it from isaacgymenvs
        self.damping_ratio = 1.0
        self.Kpos = 0.9
        self.Kori = 0.9
        self.integration_dt = 1.0
        self.gravity_compensation = True

        # Get important model IDs
        self.site_name = "attachment_site"
        self.site_id = self.model.site(self.site_name).id

        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.dof_ids = np.array(
            [self.model.joint(name).id for name in self.joint_names]
        )
        self.actuator_ids = np.array(
            [self.model.actuator(name).id for name in self.joint_names]
        )
        self.grip_actuator_id = self.model.actuator("hand").id
        self.robot_n_dofs = len(self.joint_names)

        self.mocap_name = "target"
        self.mocap_id = self.model.body(self.mocap_name).mocapid[0]

        # Controller matrices
        damping_pos = self.damping_ratio * 2 * np.sqrt(self.impedance_pos)
        damping_ori = self.damping_ratio * 2 * np.sqrt(self.impedance_ori)
        self.Kp = np.concatenate([self.impedance_pos, self.impedance_ori])
        self.Kd = np.concatenate([damping_pos, damping_ori])
        self.Kd_null = self.damping_ratio * 2 * np.sqrt(self.Kp_null)

        # Pre-allocate arrays
        self.jac = np.zeros((6, self.model.nv))
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.M_inv = np.zeros((self.model.nv, self.model.nv))
        self.robot_Mx = np.zeros((6, 6))

        # workspace bounds
        self._workspace_bounds = np.array([[0.15, 0.615], [-0.35, 0.35], [0, 0.6]])

        # Define action spaces as specified
        continuous_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        discrete_space = spaces.Discrete(2)  # 0 = open, 1 = close
        self.action_space = spaces.Tuple((continuous_space, discrete_space))

        # default obs: joint positions (7), joint velocities (7), end effector position (3), end effector orientation (4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64
        )

        # Set up rendering
        self.render_mode = render_mode
        self.viewer = None
        self.step_start = None

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset data
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.dof_ids] = self.home_joint_pos

        # Do one step to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        # Initialize target at current end effector position
        self.data.mocap_pos[self.mocap_id] = self.data.site(self.site_id).xpos.copy()
        self.data.mocap_quat[self.mocap_id] = np.array([1, 0, 0, 0])
        if False:
            mujoco.mju_mat2Quat(
                self.data.mocap_quat[self.mocap_id], self.data.site(self.site_id).xmat
            )

        # Get observation
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # unpack
        continuous_action, grip = action

        # current mocap
        current_pos = self.data.mocap_pos[self.mocap_id]
        current_quat = self.data.mocap_quat[self.mocap_id]
        print("Current Quat", current_quat)

        # delta translation
        pos_scale = 1  # 5cm maximum movement per step
        pos_delta = continuous_action[:3] * pos_scale
        new_pos = current_pos + pos_delta
        new_pos = np.clip(
            new_pos, self._workspace_bounds[:, 0], self._workspace_bounds[:, 1]
        )

        # delta orientation
        # ori_scale = 0.1  # ~6 degrees maximum rotation per step
        # ori_delta = continuous_action[3:] * ori_scale
        # TODO: calc new ori

        # update target mocap pose
        self.data.mocap_pos[self.mocap_id] = new_pos
        self.data.mocap_quat[self.mocap_id] = current_quat

        # OSC Logic Begins #

        # 1. Compute twist
        dx = self.data.mocap_pos[self.mocap_id] - self.data.site(self.site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt

        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(
            self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj
        )
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori / self.integration_dt

        # 2. Compute Jacobian
        mujoco.mj_jacSite(
            self.model, self.data, self.jac[:3], self.jac[3:], self.site_id
        )
        robot_jac = self.jac[:, : self.robot_n_dofs]

        # 3. Compute task-space inertia matrix
        mujoco.mj_solveM(self.model, self.data, self.M_inv, np.eye(self.model.nv))
        robot_M_inv = self.M_inv[: self.robot_n_dofs, : self.robot_n_dofs]
        robot_Mx_inv = robot_jac @ robot_M_inv @ robot_jac.T

        if abs(np.linalg.det(robot_Mx_inv)) >= 1e-2:
            self.robot_Mx = np.linalg.inv(robot_Mx_inv)
        else:
            self.robot_Mx = np.linalg.pinv(robot_Mx_inv, rcond=1e-2)

        # 4. Compute generalized forces
        robot_tau = (
            robot_jac.T
            @ self.robot_Mx
            @ (
                self.Kp * self.twist
                - self.Kd * (robot_jac @ self.data.qvel[self.dof_ids])
            )
        )

        # 5. Add joint task in nullspace
        robot_Jbar = robot_M_inv @ robot_jac.T @ self.robot_Mx
        robot_ddq = (
            self.Kp_null * (self.home_joint_pos - self.data.qpos[self.dof_ids])
            - self.Kd_null * self.data.qvel[self.dof_ids]
        )
        robot_tau += (
            np.eye(self.robot_n_dofs) - robot_jac.T @ robot_Jbar.T
        ) @ robot_ddq

        # 6. Add gravity compensation
        if self.gravity_compensation:
            robot_tau += self.data.qfrc_bias[self.dof_ids]

        # 7. Apply control and step
        np.clip(
            robot_tau,
            *self.model.actuator_ctrlrange[: self.robot_n_dofs, :].T,
            out=robot_tau,
        )
        self.data.ctrl[self.actuator_ids] = robot_tau

        # OSC Login Ends #

        # Actuate the gripper, tendon for franka
        self.data.ctrl[self.grip_actuator_id] = 255 if grip == 0 else 0

        # Step the simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation and info
        observation = self._get_obs()
        reward = 0.0  # Placeholder - you'll implement this
        terminated = False  # Placeholder - you'll implement this
        truncated = False  # Placeholder - you'll implement this
        info = {}  # Placeholder - you'll implement this

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render a single frame."""
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
            )

        if self.step_start is None and self.render_mode == "human":
            self.step_start = time.time()

        if self.render_mode == "human":
            self.viewer.sync()
            time_until_next_step = self.dt - (time.time() - self.step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self.step_start = time.time()

        elif self.render_mode == "rgb_array":
            # Create renderer for generating an image
            width, height = 640, 480
            renderer = mujoco.Renderer(self.model, width, height)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        # TODO: Implement this in subclass

        with mujoco.Renderer(self.model) as renderer:
            renderer.update_scene(self.data, camera="topview")
            image = renderer.render()
            print(image.shape)

        return np.zeros(self.observation_space.shape[0])

    def _get_info(self):
        # TODO: Implement this in subclass
        return {}
