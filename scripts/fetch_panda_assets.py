"""Fetch the pinned Franka Panda model used by the deterministic baseline."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
REVISION = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
MODEL_DIRECTORY = "franka_emika_panda"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "assets" / MODEL_DIRECTORY


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def fetch(force: bool = False) -> Path:
    if DESTINATION.exists():
        if not force:
            raise FileExistsError(
                f"{DESTINATION} already exists; pass --force to replace it."
            )
        shutil.rmtree(DESTINATION)

    with tempfile.TemporaryDirectory(prefix="robopilot-assets-") as temp:
        checkout = Path(temp) / "mujoco_menagerie"
        _run("git", "init", str(checkout))
        _run("git", "remote", "add", "origin", REPOSITORY, cwd=checkout)
        _run("git", "sparse-checkout", "set", MODEL_DIRECTORY, cwd=checkout)
        _run("git", "fetch", "--depth", "1", "origin", REVISION, cwd=checkout)
        _run("git", "checkout", "--detach", "FETCH_HEAD", cwd=checkout)

        source = checkout / MODEL_DIRECTORY
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            DESTINATION,
            ignore=shutil.ignore_patterns(
                "CHANGELOG.md",
                "hand.xml",
                "mjx_*.xml",
                "panda.png",
                "panda_nohand.xml",
                "scene.xml",
            ),
        )

    print(f"Installed Panda assets at {DESTINATION}")
    print(f"Source revision: {REVISION}")
    return DESTINATION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing Panda asset folder."
    )
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
