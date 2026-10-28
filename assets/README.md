# Robot assets

Robot meshes are excluded from Git. Install the deterministic baseline's pinned
Franka Panda model from Google DeepMind's MuJoCo Menagerie:

```bash
python scripts/fetch_panda_assets.py
```

The fetcher installs `franka_emika_panda/panda.xml`, its meshes, and the
upstream license. It is pinned to MuJoCo Menagerie revision
`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`.

Source: <https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/franka_emika_panda>

License: Apache-2.0. Retain the downloaded `LICENSE` file when redistributing
the model assets.
