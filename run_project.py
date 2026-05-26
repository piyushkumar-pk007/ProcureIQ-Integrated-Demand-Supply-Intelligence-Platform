from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    number: int
    name: str
    description: str
    command: List[str]


PIPELINE_STEPS: List[Step] = [
    Step(1, "data_ingestion", "Download or detect the Kaggle CSV and create clean_base.csv", [sys.executable, "-m", "src.data_ingestion"]),
    Step(2, "preprocessing", "Clean columns, parse dates, convert numerics, and save cleaned_supply_chain.csv", [sys.executable, "-m", "src.preprocessing"]),
    Step(3, "feature_engineering", "Create supply chain features and save engineered_supply_chain.csv", [sys.executable, "-m", "src.feature_engineering"]),
    Step(4, "forecasting", "Train forecasting baselines and save forecast outputs", [sys.executable, "-m", "src.forecasting"]),
    Step(5, "supplier_risk", "Build supplier risk scores and ML risk outputs", [sys.executable, "-m", "src.supplier_risk"]),
    Step(6, "optimization", "Run the procurement optimizer and save recommendations", [sys.executable, "-m", "src.optimization"]),
    Step(7, "simulation", "Run scenario simulation and save service-level and cost outputs", [sys.executable, "-m", "src.simulation"]),
]

TEST_STEP = Step(8, "tests", "Run the unit test suite", [sys.executable, "-m", "pytest"])
DASHBOARD_STEP = Step(9, "dashboard", "Launch the Streamlit dashboard", [sys.executable, "-m", "streamlit", "run", "app\\streamlit_app.py"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ProcureIQ pipeline steps in order with beginner-friendly output."
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="Start from a specific pipeline step number. Example: --from-step 4",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=7,
        help="End at a specific pipeline step number. Example: --to-step 5",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Run pytest after the pipeline steps complete.",
    )
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Launch Streamlit after the pipeline steps complete.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print available steps and exit.",
    )
    return parser.parse_args()


def print_header() -> None:
    print("\nProcureIQ Runner")
    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print("=" * 60)


def print_steps(steps: Iterable[Step]) -> None:
    for step in steps:
        print(f"{step.number}. {step.name} - {step.description}")


def select_pipeline_steps(from_step: int, to_step: int) -> List[Step]:
    if from_step < 1 or to_step > 7 or from_step > to_step:
        raise ValueError("Use a valid range such as --from-step 1 --to-step 7.")
    return [step for step in PIPELINE_STEPS if from_step <= step.number <= to_step]


def run_step(step: Step) -> None:
    print(f"\n[{step.number}] Running {step.name}")
    print(f"    {step.description}")
    print(f"    Command: {' '.join(step.command)}")
    subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)
    print(f"[{step.number}] Completed {step.name}")


def main() -> None:
    args = parse_args()
    print_header()

    if args.list_steps:
        print("Available steps:")
        print_steps(PIPELINE_STEPS + [TEST_STEP, DASHBOARD_STEP])
        return

    selected_steps = select_pipeline_steps(args.from_step, args.to_step)
    print("Pipeline steps to run:")
    print_steps(selected_steps)

    try:
        for step in selected_steps:
            run_step(step)

        if args.include_tests:
            run_step(TEST_STEP)

        if args.open_dashboard:
            print("\nLaunching dashboard. Press Ctrl+C in this window to stop Streamlit.\n")
            run_step(DASHBOARD_STEP)

        print("\nProcureIQ pipeline finished successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"\nStep failed with exit code {exc.returncode}.")
        print("Fix the error shown above, then resume with for example:")
        print(f"  {sys.executable} run_project.py --from-step {max(args.from_step, 1)}")
        raise SystemExit(exc.returncode) from exc
    except ValueError as exc:
        print(f"\nConfiguration error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

