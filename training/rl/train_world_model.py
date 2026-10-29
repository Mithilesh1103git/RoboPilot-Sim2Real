"""World-model training callable from Python, API workers, or CLI."""

import argparse

from training.rl.world_model import train_world_model


def run_training(
    dataset="training/datasets/transitions.npz",
    output="training/runs/world_model.pt",
    epochs=30,
    batch_size=256,
    seed=7,
):
    train_world_model(dataset, output, epochs, batch_size=batch_size, seed=seed)
    return {"checkpoint": str(output), "epochs": epochs, "dataset": str(dataset)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="training/datasets/transitions.npz")
    p.add_argument("--output", default="training/runs/world_model.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    result = run_training(**vars(p.parse_args(argv)))
    print(f"Saved world model to {result['checkpoint']}")


if __name__ == "__main__":
    main()
