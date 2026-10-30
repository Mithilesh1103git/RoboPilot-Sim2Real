# Planner

Nemotron high-level planning with optional, robot-agnostic retrieval context.

- `nemotron.py`: Nebius Nemotron API adapter.
- `prompts.py`: high-level planning prompt and context injection.
- `validation.py`: canonical plan parsing and validation.
- `mock.py`: test/mock planner.
- `context/`: local-first context assembly and optional Tavily provider.
- `context/schemas/`: extensible robot-knowledge data structures.

Tavily is optional. See `context/README.md` for setup, profile metadata,
knowledge extraction policy and authority boundaries.
