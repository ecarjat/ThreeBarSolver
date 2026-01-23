import argparse
from typing import Optional, Tuple

from .optimize import optimize_from_Ll
from .visualize import plot_mechanism_samples, plot_wheel_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find a mechanism configuration for a given linkage length.")
    parser.add_argument(
        "--target-bc",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Focus Bc search on the vicinity of this point.",
    )
    parser.add_argument(
        "--target-bc-radius",
        type=float,
        default=5.0,
        help="Radius around --target-bc within which to sample connector positions.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print periodic progress updates from the optimizer.",
    )
    return parser.parse_args()


def main() -> None:
    Ll = 120.0  # <-- set this
    args = parse_args()

    target_bc: Optional[Tuple[float, float]] = None
    if args.target_bc is not None:
        target_bc = (args.target_bc[0], args.target_bc[1])

    # If you want Bc closer, tighten these or set w_bc > 0
    best_d, best_m = optimize_from_Ll(
        Ll,
        n_random=1600000,
        refine_iters=600000,
        seed=2,
        # Your tightened Bc bounds (example)
        xbc_min_factor=-0.9,
        xbc_max_factor=-0.1,
        ybc_min_factor=0.1,
        ybc_max_factor=0.9,
        target_bc=target_bc,
        target_bc_radius=args.target_bc_radius,
        show_progress=args.progress,
        # Encourage compactness (set to e.g. 2.0..6.0 if needed)
        w_bc=3.0,
        # Require some minimum ROM
        min_span_deg=55.0,
        # alpha sweep
        alpha_min_deg=-160.0,
        alpha_max_deg=-20.0,
        alpha_step_deg=2.0,
    )

    if best_d is None or best_m is None:
        print("No solution found. Try widening ranges or lowering min_span_deg.")
        return

    print("Best design for Lwk =", Ll)
    print(f"  H = (0,0)")
    print(f"  Bc = ({best_d.xbc:.2f}, {best_d.ybc:.2f})")
    print(f"  Lu  (H-K)  = {best_d.Lu:.2f}")
    print(f"  Lkc (K-C)  = {best_d.Lkc:.2f}")
    print(f"  Lc  (Bc-C) = {best_d.Lc:.2f}")
    print()
    print("Motion quality:")
    print(f"  contiguous alpha span = {best_m['span_deg']:.1f} deg")
    print(f"  Wx RMS                = {best_m['wx_rms']:.2f}")
    print(f"  Wx peak-to-peak        = {best_m['wx_pp']:.2f}")
    print(f"  Wy range               = {best_m['y_range']:.2f}")
    print(f"  straightness           = {best_m['straightness']:.4f}")
    print(f"  bc_r                   = {best_m['bc_r']:.2f}")
    print(f"  score                  = {best_m['score']:.4f}")

    poses = best_m["poses"]
    plot_wheel_trajectory(poses)
    plot_mechanism_samples(poses, n=10)


if __name__ == "__main__":
    main()
