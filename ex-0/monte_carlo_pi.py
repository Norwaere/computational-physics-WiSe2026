"""
Monte Carlo estimation of Pi with corner-seeding.

This script estimates Pi using the Monte Carlo method:
1. Places 4 initial points at the corners of [0, 1] x [0, 1]: (0,0), (1,0), (0,1), (1,1).
2. Generates N - 4 uniformly distributed random points inside [0, 1] x [0, 1].
3. Checks if points lie inside the unit quarter circle: x^2 + y^2 <= 1.
   Points on the boundary (x^2 + y^2 = 1) are counted as inside.
4. Computes the ratio of inside points to total points, approximating pi / 4.
5. Repeats the process for N ranging from 10^2 to 10^7.
6. Analyzes and plots the error and convergence behavior using Matplotlib.
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_pi(n: int, rng: np.random.Generator = None) -> tuple[float, int]:
    """
    Estimate Pi using N points in the unit square [0, 1] x [0, 1].
    The first 4 points are placed at the 4 corners:
      - (0, 0): inside (0^2 + 0^2 = 0 <= 1)
      - (1, 0): inside (1^2 + 0^2 = 1 <= 1, boundary)
      - (0, 1): inside (0^2 + 1^2 = 1 <= 1, boundary)
      - (1, 1): outside (1^2 + 1^2 = 2 > 1)
    The remaining N - 4 points are drawn uniformly at random.
    Points on the circle boundary (x^2 + y^2 <= 1) are counted as inside.
    """
    if n < 4:
        raise ValueError("N must be at least 4 to place the 4 corner points.")
    if rng is None:
        rng = np.random.default_rng()

    # 4 corner points: 3 are inside or on the edge, 1 is outside
    corners_inside = 3

    n_random = n - 4
    if n_random > 0:
        # Generate random coordinates uniformly in [0, 1]
        x = rng.random(n_random)
        y = rng.random(n_random)
        # Vectorized check for points inside or on the boundary of the quarter circle
        random_inside = int(np.count_nonzero(x * x + y * y <= 1.0))
    else:
        random_inside = 0

    total_inside = corners_inside + random_inside
    pi_estimate = 4.0 * (total_inside / n)
    return pi_estimate, total_inside


def run_experiment(
    n_values: np.ndarray,
    num_trials: int = 5,
    seed: int = 42
) -> dict:
    """
    Run the Monte Carlo experiment for multiple values of N across several trials.
    """
    rng = np.random.default_rng(seed)
    
    results = {
        "n_values": n_values,
        "pi_estimates_mean": [],
        "pi_estimates_trials": [],
        "abs_errors_mean": [],
        "abs_errors_trials": [],
        "error_to_n_ratios_mean": [],
    }

    print("=" * 78)
    print(f"{'N':>10} | {'Estimated Pi':>14} | {'Abs Error':>12} | {'Rel Error (%)':>14} | {'Error / N':>12}")
    print("=" * 78)

    for n in n_values:
        trial_estimates = []
        trial_errors = []
        trial_error_to_n = []

        for _ in range(num_trials):
            pi_est, _ = monte_carlo_pi(int(n), rng=rng)
            err = abs(pi_est - math.pi)
            trial_estimates.append(pi_est)
            trial_errors.append(err)
            trial_error_to_n.append(err / n)

        mean_est = float(np.mean(trial_estimates))
        mean_err = float(np.mean(trial_errors))
        mean_err_to_n = float(np.mean(trial_error_to_n))
        rel_err_pct = (mean_err / math.pi) * 100.0

        results["pi_estimates_mean"].append(mean_est)
        results["pi_estimates_trials"].append(trial_estimates)
        results["abs_errors_mean"].append(mean_err)
        results["abs_errors_trials"].append(trial_errors)
        results["error_to_n_ratios_mean"].append(mean_err_to_n)

        print(f"{n:10d} | {mean_est:14.7f} | {mean_err:12.4e} | {rel_err_pct:13.4f}% | {mean_err_to_n:12.4e}")

    print("=" * 78)
    return results


def plot_results(results: dict, sample_n_geometry: int = 2500, save_path: str = "monte_carlo_pi.png"):
    """
    Plot comprehensive visualization including:
    1. Visual simulation layout with sample points and the 4 corner points.
    2. Pi estimate convergence towards true Pi as N increases.
    3. Absolute error vs N (log-log) compared with theoretical O(1/sqrt(N)) scaling.
    4. Error to N ratio vs N.
    """
    n_values = results["n_values"]
    abs_errors_mean = np.array(results["abs_errors_mean"])
    abs_errors_trials = np.array(results["abs_errors_trials"])
    pi_estimates_mean = np.array(results["pi_estimates_mean"])
    error_to_n_ratios = np.array(results["error_to_n_ratios_mean"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#f8f9fa")

    # -------------------------------------------------------------
    # Subplot 1: Geometric setup of the Monte Carlo Quarter Circle
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    
    # Generate sample points for demonstration
    demo_rng = np.random.default_rng(123)
    # Corner points
    corner_x = np.array([0.0, 1.0, 0.0, 1.0])
    corner_y = np.array([0.0, 0.0, 1.0, 1.0])
    corner_inside = corner_x**2 + corner_y**2 <= 1.0
    
    # Sample random points
    rand_x = demo_rng.random(sample_n_geometry - 4)
    rand_y = demo_rng.random(sample_n_geometry - 4)
    rand_inside = rand_x**2 + rand_y**2 <= 1.0

    ax1.scatter(
        rand_x[rand_inside], rand_y[rand_inside],
        s=4, color="#1f77b4", alpha=0.5, label="Inside random points"
    )
    ax1.scatter(
        rand_x[~rand_inside], rand_y[~rand_inside],
        s=4, color="#d62728", alpha=0.5, label="Outside random points"
    )
    # Highlight 4 corner points
    ax1.scatter(
        corner_x[corner_inside], corner_y[corner_inside],
        s=90, color="#2ca02c", edgecolors="black", linewidth=1.5,
        zorder=5, label="Corner points inside/edge (3)"
    )
    ax1.scatter(
        corner_x[~corner_inside], corner_y[~corner_inside],
        s=90, color="#9467bd", edgecolors="black", linewidth=1.5,
        zorder=5, label="Corner point outside (1)"
    )

    # Quarter circle boundary arc
    theta = np.linspace(0, np.pi / 2, 200)
    ax1.plot(np.cos(theta), np.sin(theta), color="black", linewidth=2.0, linestyle="--", label=r"Boundary $x^2+y^2=1$")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal", "box")
    ax1.set_title(f"Monte Carlo Setup (N = {sample_n_geometry})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x", fontsize=10)
    ax1.set_ylabel("y", fontsize=10)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Subplot 2: Pi estimate vs N
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    ax2.axhline(math.pi, color="red", linestyle="--", linewidth=1.8, label=f"True $\\pi \\approx {math.pi:.5f}$")
    ax2.plot(n_values, pi_estimates_mean, "o-", color="#1f77b4", linewidth=1.8, markersize=5, label="Mean $\\hat{\\pi}$ estimate")
    ax2.set_xscale("log")
    ax2.set_title(r"Estimation of $\pi$ vs Number of Points $N$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Points $N$ (log scale)", fontsize=10)
    ax2.set_ylabel(r"Estimated Value $\hat{\pi}$", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Subplot 3: Absolute Error vs N (Log-Log) with O(1/sqrt(N)) scaling
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#ffffff")

    # Plot individual trial errors faintly
    for trial_idx in range(abs_errors_trials.shape[1]):
        ax3.plot(
            n_values, abs_errors_trials[:, trial_idx],
            "o", color="#1f77b4", alpha=0.25, markersize=4,
            label="Single Trial" if trial_idx == 0 else None
        )

    # Plot mean absolute error
    ax3.plot(
        n_values, abs_errors_mean,
        "s-", color="#1f77b4", linewidth=2.0, markersize=6, label=r"Mean Absolute Error $|\hat{\pi} - \pi|$"
    )

    # Theoretical Monte Carlo error curve: E[Error] ~ C / sqrt(N)
    # Standard deviation of 4 * Bern(pi/4) is sqrt(pi*(4-pi)) ~= 1.642
    # Mean absolute error of standard normal is sqrt(2/pi) * sigma ~= 1.31 / sqrt(N)
    c_theory = math.sqrt(2.0 / math.pi) * math.sqrt(math.pi * (4.0 - math.pi))
    theo_curve = c_theory / np.sqrt(n_values)
    ax3.plot(
        n_values, theo_curve,
        "r--", linewidth=2.0, label=rf"Theoretical $\mathcal{{O}}(1/\sqrt{{N}})$ ($\approx {c_theory:.2f}/\sqrt{{N}}$)"
    )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title(r"Convergence: Error $|\hat{\pi} - \pi|$ vs $N$ (Log-Log)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Number of Points $N$ (log scale)", fontsize=10)
    ax3.set_ylabel(r"Absolute Error $|\hat{\pi} - \pi|$ (log scale)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, which="both", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Subplot 4: Error to N Ratio vs N (Log-Log)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#ffffff")

    ax4.plot(
        n_values, error_to_n_ratios,
        "d-", color="#2ca02c", linewidth=2.0, markersize=6, label=r"Error / $N$ ratio"
    )
    # Expected slope for Error/N is N^(-3/2)
    theo_ratio = theo_curve / n_values
    ax4.plot(
        n_values, theo_ratio,
        "k--", linewidth=1.8, label=r"Theoretical scaling $\mathcal{O}(N^{-1.5})$"
    )

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_title(r"Error-to-$N$ Ratio vs $N$ (Log-Log)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Number of Points $N$ (log scale)", fontsize=10)
    ax4.set_ylabel(r"Ratio: $|\hat{\pi} - \pi| / N$ (log scale)", fontsize=10)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nFigure successfully saved to: {save_path}")
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monte Carlo Pi Estimation")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per N value (default: 5)")
    parser.add_argument("--points", type=int, default=16, help="Number of N points between 10^2 and 10^7 (default: 16)")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive matplotlib window")
    parser.add_argument("--output", type=str, default="monte_carlo_pi.png", help="Output image file path")
    args = parser.parse_args()

    print("Starting Monte Carlo Pi Estimation Simulation...\n")
    start_time = time.time()

    # Logarithmically spaced values of N between 10^2 and 10^7
    n_values = np.unique(np.logspace(2, 7, num=args.points, dtype=np.int64))

    # Run experiment
    results = run_experiment(n_values=n_values, num_trials=args.trials, seed=42)

    total_time = time.time() - start_time
    print(f"Simulation completed in {total_time:.2f} seconds.")

    plot_results(results, sample_n_geometry=2500, save_path=args.output)
    
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
