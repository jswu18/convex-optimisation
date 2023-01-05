from typing import List

import matplotlib.pyplot as plt


def plot_loss(
    losses: List[List[float]], labels: List[str], algorithm: str, save_path: str
):
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    for i, loss in enumerate(losses):
        plt.plot(loss, label=labels[i])
    if len(labels) > 1:
        plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss ({algorithm})")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
