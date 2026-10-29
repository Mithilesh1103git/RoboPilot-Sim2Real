# Custom robot support — implementation audit

## Implemented
- Robot profiles are JSON files in `configs/robots/`; `panda.json` is the reference profile and `custom_robot.template.json` is a copy-and-edit template.
- Runtime and training each have their own profile reader (`app/robots/registry.py`, `training/robots/registry.py`). Training imports no runtime Python code.
- Runtime `/robots` and training `/api/robots` list available profiles. Both web UIs populate a robot selector from those endpoints.
- Runtime task requests accept `robot_name`; planner prompts and plan validation use profile-configured object/destination names.
- Runtime MuJoCo scene creation uses the selected robot's MJCF and profile-configured task geometry.
- Runtime IK, joint trajectories, reset/home positions and gripper commands use configured joints, end-effector body and gripper actuators instead of fixed `joint1..joint7` / `actuator8` assumptions.
- Training `RobotPickPlaceEnv` derives action/observation dimensions from the profile and supports robot selection for PPO/SAC, evaluation, SDG collection, policy rollouts and flow pipeline.
- Checkpoints carry robot/algorithm/dimension metadata; evaluation and policy rollout collection reject profile mismatches when metadata is available.
- Added profile validation utility and documentation for model asset placement.

## Requirements for adding a robot
1. Create `configs/robots/<robot_id>.json` from the template and set `id` to match the filename stem.
2. Put its MJCF and referenced assets in `assets/robots/<asset_directory>/`.
3. Copy the same model/assets to `training/assets/robots/<asset_directory>/` for the independent training application.
4. Configure ordered scalar arm joint names, home positions, end-effector body, direct joint actuators, gripper actuators/open/closed values, gripper joint positions, and task geometry.
5. Run `python -m training.robots.validate_profile --robot-name <robot_id>`, restart applications, and test with MuJoCo.

## Important compatibility boundary
The built-in controller supports MuJoCo robots whose configured scalar arm joints are driven by position actuators, and whose gripper behavior can be represented by configured actuator values. It is not a claim that arbitrary robot MJCFs will work without adaptation. Torque/velocity actuators, mobile bases, bimanual robots, specialized suction tools, unusual kinematics, and non-MuJoCo simulators need a robot-specific controller/environment adapter and a matching training action-space/reward design.

## Validation performed
- Python source compilation: passed.
- JavaScript syntax checks for both dashboards: passed.
- Test suite: 33 passed.
- Runtime `GET /robots` and training `GET /api/robots` smoke tests: HTTP 200.
- MuJoCo/Gymnasium/Stable-Baselines3 were not installed in this build environment, so model loading, physical pick-and-place, synthetic rollout execution and PPO/SAC learning were not tested end-to-end here.
