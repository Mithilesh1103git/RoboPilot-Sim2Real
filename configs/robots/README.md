# Robot profiles

Add one JSON file per robot under `configs/robots/`. The file name (without `.json`) is the `ROBOT_NAME`.

Required fields:
- `id`, `display_name`
- `asset_directory`: directory name under the runtime `assets/robots/` and training `training/assets/robots/` roots
- `mjcf_file`: MuJoCo MJCF model file in that directory
- `arm_joint_names`: ordered actuated arm joints (any count)
- `home_positions`: one safe initial joint position per arm joint
- `end_effector_body`: MuJoCo body name used for Cartesian IK
- `gripper_actuators`, `gripper_open_control`, `gripper_closed_control`: actuator names and matching control values
- `gripper_joint_positions`: optional mapping of gripper joint names to open/reset qpos values
- `task`: task geometry and object names for the default pick-and-place scene

A profile does not magically make every robot compatible. The MJCF must load in MuJoCo, have the configured joints/body/actuators, and use control semantics matching the configured values. Robots with different grippers, mobile bases, non-MJCF simulators, or task-specific kinematics may require a custom adapter/controller. Keep runtime assets under `assets/robots/<asset_directory>/` and training-owned copies under `training/assets/robots/<asset_directory>/`; do not make training import runtime Python code.

See `custom_robot.template.json` for a starting point. Do not leave template names/values unchanged; they must match the selected MJCF and actuator semantics.
