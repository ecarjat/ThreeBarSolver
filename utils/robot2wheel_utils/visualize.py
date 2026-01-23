from typing import List
import math

from .kinematics import Pose

def plot_wheel_trajectory(poses: List[Pose]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib not installed. pip install matplotlib") from e

    wx = [p.W[0] for p in poses]
    wy = [p.W[1] for p in poses]

    plt.figure()
    plt.plot(wx, wy)
    plt.axvline(0.0)
    plt.gca().invert_yaxis()  # often useful since "down" is negative y in our frame
    plt.title("Wheel trajectory W (x vs y)")
    plt.xlabel("W.x")
    plt.ylabel("W.y")
    plt.show()


def plot_mechanism_samples(poses: List[Pose], n: int = 10) -> None:
    """
    Plot a few mechanism configurations in the hip frame.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib not installed. pip install matplotlib") from e

    if not poses:
        return
    idxs = [int(i * (len(poses)-1) / max(1, n-1)) for i in range(n)]

    plt.figure()
    for i in idxs:
        p = poses[i]
        # segments: H-K, K-C, K-W, Bc-C
        H, K, C, W, Bc = p.H, p.K, p.C, p.W, p.Bc
        plt.plot([H[0], K[0]], [H[1], K[1]])
        plt.plot([K[0], C[0]], [K[1], C[1]])
        plt.plot([K[0], W[0]], [K[1], W[1]])
        plt.plot([Bc[0], C[0]], [Bc[1], C[1]])

    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.gca().invert_yaxis()
    plt.title("Mechanism samples (multiple alphas)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
