import os
import time

import mujoco.renderer
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces


class FrankaEnv(MujocoEnv):
    def __init__(self, model_path=None, render_mode=None):

        IMAGE_SHAPE = (224, 224, 3)

        observation_space = spaces.Dict(
            {
                "joint_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
                ),
                "eef_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "eef_quat": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),
                "gripper_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                ),
                "front_view": spaces.Box(
                    low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8
                ),
                "top_view": spaces.Box(
                    low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8
                ),
                "right_view": spaces.Box(
                    low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8
                ),
                "left_view": spaces.Box(
                    low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8
                ),
            }
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
        )

        # default dt of model is 0.002 which means 500Hz
        self.gym_step = 100

        # Override with mujoco viewer
        self.mujoco_renderer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )

        # NOTE: make the camera look like the agentview
        # TODO: can finetune to make it better
        self.mujoco_renderer.cam.distance = 2.05
        self.mujoco_renderer.cam.azimuth = 180
        self.mujoco_renderer.cam.elevation = -25
        self.mujoco_renderer.cam.lookat[:] = np.array([-0.5, -0.0, 0.0])

        self.step_start = None

        self._init_osc_params()
        self._init_offscreen_renderer()

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
        self.finger_joint_names = ["finger_joint1", "finger_joint2"]
        self.scene_cam_names = ["front_view", "top_view", "left_view", "right_view"]
        self.dof_ids = np.array(
            [self.model.joint(name).id for name in self.joint_names]
        )
        self.finger_dof_ids = np.array(
            [self.model.joint(name).id for name in self.finger_joint_names]
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

    def render(self, mode="human"):
        """
        only meant for human mode
        """
        assert mode == "human", "Only meant for human mode"
        self._render_frame()

    def _render_frame(self):
        """
        Render the frame for human mode
        """

        if self.step_start is None:
            self.step_start = time.time()

        self.mujoco_renderer.sync()
        time_until_next_step = self.dt - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.step_start = time.time()

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

    def _init_offscreen_renderer(self):
        self.offscreen_renderer = mujoco.renderer.Renderer(
            model=self.model, height=224, width=224
        )

    def reset_model(self):
        self.data.qpos[self.dof_ids] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):

        _obs_dict = {}

        ee_quat = np.zeros(4)  # 4
        mujoco.mju_mat2Quat(ee_quat, self.data.site(self.site_id).xmat)

        _obs_dict["joint_pos"] = self.data.qpos[self.dof_ids]
        # TODO: if needed add joint velocity, using "self.data.qvel[self.dof_ids]"
        _obs_dict["eef_pos"] = self.data.site(self.site_id).xpos
        _obs_dict["eef_quat"] = ee_quat

        # gripper position
        _obs_dict["gripper_pos"] = np.array(
            [pos for pos in self.data.qpos[self.finger_dof_ids]]
        )

        # images fromc cameras in the scene
        for cam_name in self.scene_cam_names:
            self.offscreen_renderer.update_scene(
                data=self.data, camera=self.model.camera(cam_name).id
            )
            _obs_dict[cam_name] = self.offscreen_renderer.render()

        return _obs_dict

    def step(self, action, mode="delta"):

        assert mode in ["delta", "abs"], f"Invalid mode {mode}"

        # NOTE: grip will remain same for either mode

        for i in range(self.gym_step):
            continuous_action, grip = action

            if mode == "delta":
                self._compute_new_target(continuous_action)

            else:
                self.data.mocap_pos[self.mocap_id] = continuous_action[:3]

                # NOTE: for now orientation is unchanged
                if False:
                    _target_quat = np.zeros(4)
                    mujoco.mju_euler2Quat(_target_quat, continuous_action[3:])
                    self.data.mocap_quat[self.mocap_id] = _target_quat

            robot_tau = self._compute_osc()
            ctrl = np.zeros(self.model.nu)
            ctrl[self.actuator_ids] = robot_tau
            ctrl[self.grip_actuator_id] = 255 if grip == 0 else 0
            self.do_simulation(ctrl, self.frame_skip)
            if self.render_mode == "human":
                self.render()

        obs = self._get_obs()
        reward = 0.0  # Customize as needed
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def _compute_new_target(self, continuous_action):
        """
        meant to compute updated pose for the mocap(target), for delta mode
        """

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

        # NOTE: for now orientation is unchanged
        if False:
            # delta orientation
            ori_scale = 0.1  # ~6 degrees maximum rotation per step
            ori_delta = continuous_action[3:] * ori_scale

        # update target mocap pose
        self.data.mocap_pos[self.mocap_id] = new_pos
        self.data.mocap_quat[self.mocap_id] = current_quat

    def _compute_osc(self):
        """
        Logic for OSC, assumes that the target pose is the mocap pose
        """

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

        return robot_tau
