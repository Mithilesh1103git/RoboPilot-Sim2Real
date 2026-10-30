# Planner Context Subsystem

Robot-agnostic, local-first retrieval-augmented context for Nemotron.

## Optional Tavily setup

Install the optional provider dependency:

```bash
pip install tavily-python
```

Configure `.env`:

```dotenv
TAVILY_ENABLED=false
TAVILY_API_KEY=
TAVILY_TIMEOUT_SECONDS=15
TAVILY_MAX_RESULTS=5
PLANNER_CONTEXT_CACHE_TTL_SECONDS=86400
```

Tavily runs only when `TAVILY_ENABLED=true` and the selected robot profile has
`robot_context.search_enabled=true`. If disabled or unavailable, planning
continues with local profile and simulator observations.

## Robot profile metadata

Example for a Franka Panda profile:

```json
{
  "robot_context": {
    "manufacturer": "Franka Robotics",
    "model": "Franka Emika Panda",
    "robot_category": "manipulator",
    "search_enabled": true,
    "search_focus": "payload, reach, kinematics, end-effector, sensing and safety",
    "preferred_domains": ["franka.de"]
  }
}
```

This is metadata for retrieval, not a replacement for the actual robot profile
or loaded MJCF. Other robot families can use different categories and search
focus. Do not hardcode Panda-specific attributes into the generic extractor.

## Knowledge extraction policy

The current implementation normalizes and preserves source evidence, URLs,
titles, retrieval scores and evidence IDs. It deliberately does not invent
structured facts from snippets. A structured fact extractor can later populate
`robot_knowledge.categories` and `robot_knowledge.facts`, using evidence IDs
for provenance. Categories are extensible and optional.

## Authority boundaries

- Local profile and loaded MuJoCo model are authoritative for the simulated robot.
- External evidence is supplementary, untrusted and model-specific.
- Missing or conflicting facts remain unknown.
- Web specifications cannot override joint limits, actuator ranges, geometry,
  scene state, controller constraints or execution outcomes.
- Nemotron proposes high-level intent; plan validation and deterministic
  controller execution remain separate.
