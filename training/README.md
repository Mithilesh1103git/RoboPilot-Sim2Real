# Independent training project

This package owns its simulator scene, MuJoCo loading, observations, action space, reward, synthetic transitions, RL trainer, evaluation, and artifacts. It must not import `app/`, call runtime API endpoints, or use runtime services.

No LLM is used for data generation, rewards, labels, demonstrations, or policy learning. Transitions come from simulator interaction and rewards are numeric functions of state.

The Panda environment is a baseline continuous joint-target task. Do not claim learned grasping success without running evaluation and inspecting results.
