import argparse

from training.rl.world_model import train_world_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="training/datasets/transitions.npz")
    p.add_argument("--output", default="training/runs/world_model.pt")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    train_world_model(a.dataset, a.output, a.epochs)


if __name__ == "__main__":
    main()
