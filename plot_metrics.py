import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("system_mean_speed", "System Mean Speed (m/s)"),
    ("system_total_waiting_time", "System Total Waiting Time (s)"),
    ("system_total_stopped", "System Total Stopped Vehicles"),
]


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    df.rename(columns=lambda col: col.strip(), inplace=True)
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
    return df


def plot_comparison(
    baseline_df: pd.DataFrame,
    orchestrator_df: pd.DataFrame,
    labels: Dict[str, str],
    output_path: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 12), sharex=True)
    step_col = "step"

    for ax, (metric, title) in zip(axes, METRICS):
        if metric not in baseline_df.columns and metric not in orchestrator_df.columns:
            ax.set_title(f"{title} (missing in both CSVs)")
            continue

        if metric in baseline_df.columns:
            ax.plot(
                baseline_df[step_col],
                baseline_df[metric],
                label=labels["baseline"],
                color="#1f77b4",
            )
        if metric in orchestrator_df.columns:
            ax.plot(
                orchestrator_df[step_col],
                orchestrator_df[metric],
                label=labels["orchestrator"],
                color="#d62728",
            )

        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Simulation Step (s)")
    fig.tight_layout()

    if output_path:
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SUMO metrics between CSV runs.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="./datos_baseline.csv",
        help="Path to baseline CSV (with columns step/system_*).",
    )
    parser.add_argument(
        "--orchestrator",
        type=str,
        default="./metrics/orchestrator_eval_conn1_ep1.csv",
        help="Path to orchestrator CSV exported by the RL pipeline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./metrics/comparison.png",
        help="Optional path to save the generated plot (PNG).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot instead of saving to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_df = load_csv(args.baseline)
    orchestrator_df = load_csv(args.orchestrator)

    labels = {"baseline": "Baseline", "orchestrator": "Orchestrator"}
    output_path = None if args.show else args.output

    plot_comparison(baseline_df, orchestrator_df, labels, output_path)


if __name__ == "__main__":
    main()
