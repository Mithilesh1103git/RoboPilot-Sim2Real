import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# Point to your local UR5e xml file
LOCAL_ASSETS_PATH = (
    r"C:\Users\mithi_w\mydata\git_repos\personal_repos\RoboPilot-Sim2Real\assets\robots"
)
MODEL_PATH = os.path.abspath(
    os.path.join(
        LOCAL_ASSETS_PATH, "universal_robots_ur5e", "ur5e_table_tray_glass.xml"
    )
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

print(f"Loading UR5e model from: {MODEL_PATH}")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 2. Define Fixed Baseline Initial Conditions (Exactly 6 joints for UR5e)
# HOME_JOINTS = np.array([0.0, -1.2, 1.7, -2.1, -1.6, 0.0])
HOME_JOINTS = np.array([0.0, -1.2, 0.5, -0.1, -0.6, 0.5])

# Fetch IDs for free-moving objects to lock their baseline positions
glass_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "glass")
tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")

# Fixed starting coordinates [X, Y, Z]
FIXED_TRAY_POS = np.array([-0.4, 0.0, 0.405])
FIXED_GLASS_POS = np.array([-0.4, 0.0, 0.47])

# Track steps since the last reset to calculate a clean phase for the sine wave
steps_since_reset = 0


def reset_to_baseline(model, data):
    """Resets the simulation and forces every entity to its fixed baseline position."""
    global steps_since_reset
    mujoco.mj_resetData(model, data)
    steps_since_reset = 0  # Reset the phase clock to zero for smooth movement

    # 1. Set Robot Baseline Joints (6 Degrees of Freedom)
    data.qpos[0:6] = HOME_JOINTS

    # 2. FIX: Set Free Objects Baseline directly by body name IDs safely
    # If your XML doesn't use freejoints, we assign values using data.qpos indexing safely:
    try:
        glass_qpos_adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "glass_joint")
        ]
        # Check if the joint space allows a 3D coordinate vector assignment
        data.qpos[glass_qpos_adr : glass_qpos_adr + 3] = FIXED_GLASS_POS
    except ValueError:
        # Fallback if the joint is a 1-DOF joint
        data.qpos[glass_qpos_adr] = FIXED_GLASS_POS[0]  # Assign single axis fallback

    try:
        tray_qpos_adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tray_joint")
        ]
        data.qpos[tray_qpos_adr : tray_qpos_adr + 3] = FIXED_TRAY_POS
    except ValueError:
        data.qpos[tray_qpos_adr] = FIXED_TRAY_POS[0]

    # Force an immediate internal physics update
    mujoco.mj_forward(model, data)
    print("\n--- Simulation Reset to Baseline ---")
    print(f"Robot Joint 0 Position : {data.qpos[0]:.4f}")
    print(f"Glass Body Position (XYZ): {data.xpos[glass_body_id]}")
    print(f"Tray Body Position (XYZ) : {data.xpos[tray_body_id]}")


def control_callback(model, data):
    global steps_since_reset
    # Calculate a dedicated trajectory time that always starts at 0 after a reset
    sim_time = steps_since_reset * model.opt.timestep

    # Create a copy of the home joints to modify safely (6 elements)
    target_joints = HOME_JOINTS.copy()

    # Generate 6 random amplitudes cleanly in one go
    random_scales = np.random.uniform(-0.1, 0.9, size=6)

    # Dynamically sway all 6 joints using a sine wave
    target_joints[0] += random_scales[0] * np.sin(sim_time * 1.5)
    target_joints[1] += random_scales[1] * -np.sin(sim_time * 1.5)
    target_joints[2] += random_scales[2] * np.cos(sim_time * 1.5)
    target_joints[3] += 1.5 * np.sin(
        sim_time * 1.5
    )  # Heavy oscillation on the elbow joint
    target_joints[4] += random_scales[4] * np.cos(sim_time * 1.5)
    target_joints[5] += random_scales[5] * np.cos(sim_time * 1.5)

    # FIX: Write exactly 6 values to the 6 robot joint actuators
    data.ctrl[0:6] = target_joints


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
    viewer.cam.lookat = [-0.4, 0.0, 0.45]  # [X, Y, Z] Focus center of the table
    viewer.cam.distance = 2.0  # Zoomed closer to the robot base
    viewer.cam.elevation = -45.0  # Angle looking down (degrees)
    viewer.cam.azimuth = 110.0  # Horizontal rotation angle (degrees)

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
