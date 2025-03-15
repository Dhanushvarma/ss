import mujoco
import mujoco.viewer
import numpy as np
import time

# reference joint config
home_joint_pos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]  # except the hand joints

# Cartesian impedance control gains.
impedance_pos = np.asarray([500.0, 500.0, 500.0])  # [N/m]
impedance_ori = np.asarray([250.0, 250.0, 250.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([10.0]*7)  # NOTE: got it from isaacgymenvs

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.9

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.9

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.001


def main() -> None:

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(
        "/home/dhanush/dhanush_ws/lira/ss/assets/franka_emika_panda/scene.xml"
    )

    data = mujoco.MjData(model)

    model.opt.timestep = dt

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    grip_actuator_id = model.actuator("hand").id
    robot_n_dofs = len(joint_names)

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    robot_Mx = np.zeros((6, 6))

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetData(model, data)
        data.qpos[dof_ids] = home_joint_pos

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # TODO: revert
        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            robot_jac = jac[:, :robot_n_dofs]

            # Compute the task-space inertia matrix.
            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            robot_M_inv = M_inv[:robot_n_dofs, :robot_n_dofs]
            robot_Mx_inv = robot_jac @ robot_M_inv @ robot_jac.T
            if abs(np.linalg.det(robot_Mx_inv)) >= 1e-2:
                robot_Mx = np.linalg.inv(robot_Mx_inv)
            else:
                robot_Mx = np.linalg.pinv(robot_Mx_inv, rcond=1e-2)

            # Compute generalized forces.
            robot_tau = (
                robot_jac.T
                @ robot_Mx
                @ (Kp * twist - Kd * (robot_jac @ data.qvel[dof_ids]))
            )

            # Add joint task in nullspace.
            robot_Jbar = robot_M_inv @ robot_jac.T @ robot_Mx
            robot_ddq = (
                Kp_null * (home_joint_pos - data.qpos[dof_ids])
                - Kd_null * data.qvel[dof_ids]
            )
            robot_tau += (np.eye(robot_n_dofs) - robot_jac.T @ robot_Jbar.T) @ robot_ddq

            # Add gravity compensation.
            if gravity_compensation:
                robot_tau += data.qfrc_bias[dof_ids]

            # Set the control signal and step the simulation.
            # np.clip(robot_tau, *model.actuator_ctrlrange.T, out=robot_tau)
            np.clip(robot_tau, *model.actuator_ctrlrange[:robot_n_dofs,:].T, out=robot_tau)
            data.ctrl[actuator_ids] = robot_tau[actuator_ids]
            data.ctrl[grip_actuator_id] = 125  # NOTE: 0 means open and 255 means close
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
