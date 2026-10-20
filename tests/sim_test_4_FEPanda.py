import os
import time

import mujoco
import mujoco.viewer
import numpy as np

# Point to your local panda.xml file
LOCAL_ASSETS_PATH = r"C:\Users\mithi_w\mydata\git_repos\personal_repos\RoboPilot\assets"
MODEL_PATH = os.path.abspath(
    os.path.join(LOCAL_ASSETS_PATH, "franka_emika_panda", "panda_table_glass.xml")
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading Franka Panda model from: {MODEL_PATH}")

# MuJoCo will resolve the paths relative to the XML file location automatically.
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 2. Define Fixed Baseline Initial Conditions
HOME_JOINTS = np.array([0.0, -0.7, 0.0, -2.5, 0.0, 2.1, 0.78])

# Fetch IDs for free-moving objects to lock their baseline positions
glass_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "glass")
tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")

# Fixed starting coordinates [X, Y, Z]
FIXED_TRAY_POS = np.array([0.4, 0.0, 0.405])
FIXED_GLASS_POS = np.array([0.4, 0.0, 0.47])

# Track steps since the last reset to calculate a clean phase for the sine wave
steps_since_reset = 0


def reset_to_baseline(model, data):
    """Resets the simulation and forces every entity to its fixed baseline position."""
    global steps_since_reset
    mujoco.mj_resetData(model, data)
    steps_since_reset = 0  # Reset the phase clock to zero for smooth movement

    # 1. Set Robot Baseline Joints
    data.qpos[0:7] = HOME_JOINTS

    # 2. Set Free Objects Baseline (qpos for free joints is [X, Y, Z, Qw, Qx, Qy, Qz])
    glass_qpos_adr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "glass_joint")
    ]
    tray_qpos_adr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tray_joint")
    ]

    data.qpos[tray_qpos_adr : tray_qpos_adr + 3] = FIXED_TRAY_POS
    data.qpos[glass_qpos_adr : glass_qpos_adr + 3] = FIXED_GLASS_POS

    # Forward kinematics to update body positions globally
    mujoco.mj_forward(model, data)
    print("\n--- Simulation Reset to Baseline ---")
    print(f"Robot Joint 0 Position : {data.qpos[0]:.4f}")
    print(f"Glass Body Position (XYZ): {data.xpos[glass_body_id]}")
    print(f"Tray Body Position (XYZ) : {data.xpos[tray_body_id]}")


def control_callback(model, data):
    # Calculate a dedicated trajectory time that always starts at 0 after a reset
    sim_time = steps_since_reset * model.opt.timestep

    # Create a copy of the home joints to modify safely
    target_joints = HOME_JOINTS.copy()

    # Dynamically sway Joint 0 (base) and Joint 3 (elbow) using a sine wave
    target_joints[0] += np.random.rand(1, 1)[0][0].item() * np.sin(sim_time * 1.5)
    target_joints[1] += np.random.rand(1, 1)[0][0].item() * np.sin(
        sim_time * 1.5
    )  # Amplitude of 0.5 rad, frequency 2.0
    target_joints[2] += np.random.rand(1, 1)[0][0].item() * np.cos(sim_time * 1.5)
    # target_joints[3] += np.random.rand(1, 1)[0][0].item() * np.cos(sim_time * 1.5)
    target_joints[3] += 1.2 * np.cos(sim_time * 1.5)
    target_joints[4] += np.random.rand(1, 1)[0][0].item() * np.cos(sim_time * 1.5)
    target_joints[5] += np.random.rand(1, 1)[0][0].item() * np.cos(sim_time * 1.5)
    target_joints[6] += np.random.rand(1, 1)[0][0].item() * np.cos(sim_time * 1.5)

    # Send the dynamically changing joints to the robot actuators
    data.ctrl[0:7] = target_joints
    gripper_close_amount = 0.02 + 0.2 * np.sin(sim_time * 3.0)
    data.ctrl[7] = gripper_close_amount

    print(f"Control values: {data.ctrl}")


# Register the corrected controller
mujoco.set_mjcb_control(control_callback)

# 4. Run Dynamic Verification Loop
print("Launching Verification Run...")
reset_to_baseline(model, data)

step_count = 0
reset_count = 0
max_retries = 5
enable_local_viewer: bool = True

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Configure the internal camera object directly
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Free-flying camera mode
    viewer.cam.lookat = [0.4, 0.0, 0.45]  # [X, Y, Z] Focus center of the table
    viewer.cam.distance = 3.0  # Zoom distance
    viewer.cam.elevation = -25.0  # Angle looking down (degrees)
    viewer.cam.azimuth = 135.0  # Horizontal rotation angle (degrees)

    while viewer.is_running() and enable_local_viewer and reset_count < max_retries:
        step_start = time.time()

        # Step the physics engine forward
        mujoco.mj_step(model, data)
        viewer.sync()

        step_count += 1
        steps_since_reset += 1  # Drive the trajectory controller clock forward

        # Trigger a reset every 500 simulation steps (~1 second of sim time)
        if step_count >= 500:
            reset_to_baseline(model, data)
            reset_count += 1
            step_count = 0

        # Keep rendering frame rates aligned with real-world time steps
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print(
    "\n[SUCCESS] Baseline verification complete. Check your terminal output above to verify matching coordinates across both resets."
)

# # 3. Simple controller for baseline testing
# def control_callback(model, data):
#     # Hold the home position firmly during the baseline check
#     data.ctrl[0:7] = HOME_JOINTS
#     data.ctrl = 255.0
