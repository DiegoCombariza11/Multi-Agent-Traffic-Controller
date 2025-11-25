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
    evaluation_df: Optional[pd.DataFrame],
    labels: Dict[str, str],
    output_path: Optional[str] = None,
) -> None:
    step_col = "step"
    
    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        if metric in baseline_df.columns:
            ax.plot(
                baseline_df[step_col],
                baseline_df[metric],
                label=labels["baseline"],
                color="#1f77b4",
                linewidth=2,
            )
            has_data = True
        if metric in orchestrator_df.columns:
            ax.plot(
                orchestrator_df[step_col],
                orchestrator_df[metric],
                label=labels["regional_agents"],
                color="#d62728",
                linewidth=2,
            )
            has_data = True
        if evaluation_df is not None and metric in evaluation_df.columns:
            ax.plot(
                evaluation_df[step_col],
                evaluation_df[metric],
                label=labels["local_agents"],
                color="#2ca02c",
                linewidth=2,
            )
            has_data = True

        if not has_data:
            print(f"Warning: {metric} missing in all CSVs, skipping plot.")
            plt.close(fig)
            continue

        ax.set_xlabel("Simulation Step (s)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()

        if output_path:
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)
            # Generate individual filename for each metric
            base_name = os.path.splitext(output_path)[0]
            ext = os.path.splitext(output_path)[1] or ".png"
            metric_output = f"{base_name}_{metric}{ext}"
            fig.savefig(metric_output, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {metric_output}")
            plt.close(fig)
        else:
            plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SUMO metrics between CSV runs.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="./metrics/datos_baseline.csv",
        help="Path to baseline CSV (with columns step/system_*).",
    )
    parser.add_argument(
        "--regional-agents",
        type=str,
        default="./metrics/orchestrator_eval_conn1_ep1.csv",
        help="Path to orchestrator CSV exported by the RL pipeline.",
    )
    parser.add_argument(
        "--local-agents",
        type=str,
        default="./graphics/datos_IA_evaluacion_conn1_ep1.csv",
        help="Optional evaluation CSV to overlay (set to '' to skip).",
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
    regional_agents_df = load_csv(args.regional_agents)

    local_agents_df = None
    if args.local_agents:
        local_agents_df = load_csv(args.local_agents)
    labels = {"baseline": "Baseline", "regional_agents": "Regional Agents"}
    if local_agents_df is not None:
        labels["local_agents"] = "Local Agents"
    output_path = None if args.show else args.output

    plot_comparison(baseline_df, regional_agents_df, local_agents_df, labels, output_path)

if __name__ == "__main__":
    main()
