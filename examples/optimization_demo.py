"""
Optimization Demo Script

Full end-to-end demonstration of SEDA's built-in Optuna integration.

What it shows:
- Hyperparameter search for SMACrossoverStrategy
- Data pre-loading for massive speed gains on repeated trials
- Sophisticated composite objective (Calmar × Sharpe² with low-sample penalty)
- Automatic best-trial backtest + rich HTML report
- Optuna visualisation plots (history, importance, contours)

Run this script directly:
    python examples/optimization_demo.py
"""

from datetime import datetime
import os
import warnings
import numpy as np

from src.backtester.core.backtester import Backtester
from src.backtester.strategies.sma_crossover import SMACrossoverStrategy
from src.backtester.sizers.sizers import FixedNotionalSizer
from src.backtester.optimization.optimizer import Optimizer
from src.backtester.reporting.reporter import Reporter

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ------------------------------------------------------------------
# Configuration — easy to tweak for different assets/timeframes
# ------------------------------------------------------------------
config = {
    "assets": ["BTCUSDT"],
    "timeframes": ["1h"],
    "start_date": datetime(2023, 1, 1),
    "end_date": datetime(2024, 12, 1),
    "initial_capital": 100_000.0,
    "leverage_cap": 5.0,  # Realistic perp leverage cap
    "n_trials": 10,  # Increase for serious optimisation
    "timeout": None,  # Seconds; None = run all trials
    "pre_load_data": True,  # Huge speedup — data loaded once
    "objective_metric": "calmar_only",  # Options: calmar_sharpe, sqn, calmar_only, pnl_dd_r2
    "min_trades_for_optimization": 10,  # Penalise overly sparse strategies
    "study_name": "SMA_Crossover_Opt_2025",
}

# ------------------------------------------------------------------
# Run the optimisation
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Create optimizer instance
    optimizer = Optimizer(
        config=config,
        strategy_class=SMACrossoverStrategy,
        timeout=config["timeout"],
        report_path=f"../reports/opt_best_{config['study_name']}.html"
    )

    # Execute study — prints progress and saves to in-memory storage
    print(f"\nStarting Optuna optimisation: {config['n_trials']} trials")
    study = optimizer.optimize()

    # ------------------------------------------------------------------
    # Summary of results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION COMPLETE")
    print("=" * 80)
    print(f"Best trial value: {study.best_value:,.2f}")
    print(f"Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Total completed trials: {len(study.trials)}")
    print(
        f"Best trial HTML report saved to: ../reports/opt_best_{config['study_name']}.html"
    )

    # ------------------------------------------------------------------
    # Optional: show Optuna visualisation plots
    # ------------------------------------------------------------------
    try:
        import optuna.visualization as vis
        import matplotlib.pyplot as plt

        print("\nGenerating Optuna visualisation plots...")

        fig1 = vis.plot_optimization_history(study)
        fig1.write_image("../reports/opt_history.png")

        fig2 = vis.plot_param_importances(study)
        fig2.write_image("../reports/opt_importance.png")

        fig3 = vis.plot_contour(study, params=["fast_period", "slow_period"])
        fig3.write_image("../reports/opt_contour.png")

        print("Visualisation images saved to reports/ directory")
    except Exception as e:
        print("\nCould not generate all Optuna visualizations.")
        print("Ensure optional dependencies are installed:")
        print("    pip install \"seda-backtester[optimization]\"")
        print(f"(Error: {e})")

    print(
        "\nDemo complete — open the generated HTML report to see the best strategy performance!"
    )
