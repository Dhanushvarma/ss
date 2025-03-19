import os

import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer


class FrankaEnv(MujocoEnv):
    def __init__(self, model_path=None, render_mode=None):
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # default model path
        if model_path is None:
            pwd = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(pwd, "../../assets/franka_emika_panda/scene.xml")

        super().__init__(
            model_path=model_path,
            frame_skip=1,
            observation_space=observation_space,
            render_mode=render_mode,
            width=480,
            height=480,
        )

        self._init_osc_params()
        self._init_other_renders()

        # Franks ID's
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

        # Attachment Site
        self.site_name = "attachment_site"
        self.site_id = self.model.site(self.site_name).id

        # Franks actuator & joints
        self.grip_actuator_id = self.model.actuator("hand").id
        self.robot_n_dofs = len(self.joint_names)

        # mocap, used as target
        self.mocap_name = "target"
        self.mocap_id = self.model.body(self.mocap_name).mocapid[0]

        # reset joint position for the Franka Arm
        self.home_joint_pos = [
            0,
            0,
            0,
            -1.57079,
            0,
            1.57079,
            -2.29,
        ]  # except the hand joints

        # action Space
        self.action_space = spaces.Tuple(
            (
                spaces.Box(low=-0.1, high=0.1, shape=(6,), dtype=np.float32),
                spaces.Discrete(2),
            )
        )

    def _init_osc_params(self):
        """
        All the OSC related parameters
        """

        # osc parameters
        self.impedance_pos = np.array([500.0, 500.0, 500.0])  # [N/m]
        self.impedance_ori = np.array([250.0, 250.0, 250.0])  # [Nm/rad]
        self.Kp_null = np.array([10.0] * 7)  # NOTE: got it from isaacgymenvs
        self.damping_ratio = 1.0
        self.Kpos = 0.9
        self.Kori = 0.9
        self.integration_dt = 1.0
        self.gravity_compensation = True

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

    def _init_other_renders(self):

        self.offscreen_renderer = OffScreenViewer(
            model=self.model, data=self.data, width=224, height=224
        )

    def reset_model(self):
        self.data.qpos[self.dof_ids] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        joint_pos = self.data.qpos[self.dof_ids]
        joint_vel = self.data.qvel[self.dof_ids]
        ee_pos = self.data.site(self.site_id).xpos
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, self.data.site(self.site_id).xmat)

        # below code works, but acts weird.
        if False:
            _camera_id = self.model.camera("topview").id
            rgb = self.offscreen_renderer.render(
                render_mode="rgb_array", camera_id=_camera_id
            )

        return np.concatenate([joint_pos, joint_vel, ee_pos, ee_quat])

    def step(self, action):
        continuous_action, grip = action
        robot_tau = self._compute_osc(continuous_action)
        ctrl = np.zeros(self.model.nu)
        ctrl[self.actuator_ids] = robot_tau
        ctrl[self.grip_actuator_id] = 255 if grip == 0 else 0
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        reward = 0.0  # Customize as needed
        terminated = False
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _compute_osc(self, continuous_action):

        # current mocap
        current_pos = self.data.mocap_pos[self.mocap_id]
        current_quat = self.data.mocap_quat[self.mocap_id]

        # delta translation
        pos_scale = 1
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

        ### NOTE : OSC Logic Begins ###

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

        # 7. Clip to actuator limits
        np.clip(
            robot_tau,
            *self.model.actuator_ctrlrange[: self.robot_n_dofs, :].T,
            out=robot_tau,
        )

        ### NOTE : OSC Logic Ends ###

        return robot_tau
