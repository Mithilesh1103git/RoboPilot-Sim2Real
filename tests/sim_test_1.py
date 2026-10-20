import time

import mujoco.viewer

# 1. Define your model XML as a string
xml_string = """
<mujoco>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid_mat" texture="grid" texrepeat="2 2" texuniform="true"/>
    </asset>
    <worldbody>
        <light pos="0 0 2"/>
        <geom type="plane" size="5 5 0.1" material="grid_mat"/>
        <body name="box" pos="0 0 1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1" mass="1"/>
        </body>
    </worldbody>
</mujoco>
"""

# 2. Build the assets parameter dictionary (maps simulated filenames to raw bytes)
# (Note: In this example the XML generates procedural assets, but you can inject raw file bytes here)
assets_dict = {}

# 3. Load the model from the string using the assets parameter
model = mujoco.MjModel.from_xml_string(xml_string, assets=assets_dict)
data = mujoco.MjData(model)

# 4. Set your custom step multiplier
N_STEPS = 5

# Launch passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Step the physics simulation N times per render frame loop
        mujoco.mj_step(model, data, nstep=N_STEPS)

        # Sync viewer with the latest simulation state
        viewer.sync()

        # Limit frame rate to match real-time (accounting for N physics steps taken)
        time_until_next_step = (model.opt.timestep * N_STEPS) - (
            time.time() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
