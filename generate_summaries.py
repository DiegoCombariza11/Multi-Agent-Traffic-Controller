"""
Generate edge_summary.json and agents_summary.json for LLMReporter.

Usage:
    python generate_summaries.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("Generating summaries for LLMReporter")
    print("=" * 60)
    
    # Check if edgeData.xml exists
    edge_data = Path("./sumoData/edgeData.xml")
    if not edge_data.exists():
        print(f"\nError: {edge_data} not found!")
        print("Run the simulation first: python ./agents/Agents_orchestator.py")
        sys.exit(1)
    
    # Check if actions_history.csv exists
    actions_history = Path("./actions_history.csv")
    if not actions_history.exists():
        print(f"\nWarning: {actions_history} not found!")
      

    if actions_history.exists():
        print("\n[2/2] Parsing actions_history.csv to agents_summary.json...")
        result = subprocess.run(
            [
                sys.executable,
                "./LLMReporter/services/parse_actions_history.py",
                "--actions-history", "./actions_history.csv",
                "--output", "./LLMReporter/core/agents_summary.json",
            ],
            cwd=Path(__file__).parent,
        )
        
        if result.returncode != 0:
            print("\nActions history parsing failed!")
            sys.exit(1)
    else:
        print("\n[2/2] Skipping agents_summary (no actions_history.csv)")
    
    print("\n" + "=" * 60)
    print("Summaries generated successfully")
    print("=" * 60)
    print("\nLLMReporter is ready to use:")
    print("  python ./LLMReporter/reporter_cli.py")


if __name__ == "__main__":
    main()
