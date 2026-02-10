"""Optuna-based hyperparameter optimization with efficient data pre-loading.

Optimizes a single strategy class across configurable trials/objectives.
Pre-loads data once for massive speedups on multi-asset/multi-TF runs.
Automatically runs and reports the best parameters.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import linregress

from backtester.core.engine import Backtester
from backtester.reporting.reporter import Reporter
from backtester.core.results import BacktestResults


class Optimizer:
    """Hyperparameter optimizer for a single strategy class."""

    def __init__(
        self,
        config: Dict[str, Any],
        strategy_class: type,
        timeout: Optional[int] = None,
        report_path: str = None
    ) -> None:
        self.config = config
        self.strategy_class = strategy_class
        self.n_trials = config["n_trials"]
        self.timeout = timeout or config.get("timeout")

        # Data pre-loading for speed
        self.pre_load = config.get("pre_load_data", True)
        self.pre_arrays: Dict[tuple[str, str], dict] = {}

        if self.pre_load:
            # Dummy strategy to trigger data loading (no leverage during preload)
            dummy_params = strategy_class.Params()
            dummy_strat = strategy_class(name="preload_dummy", params=dummy_params)

            dummy_bt = Backtester(
                timeframes=config["timeframes"],
                asset_list=config["assets"],
                strategies=[dummy_strat],
                start_date=config["start_date"],
                end_date=config["end_date"],
                initial_capital=config.get("initial_capital", 150_000.0),
                leverage=None,  # No liquidation during preload
            )
            dummy_bt.load_data()  # Replace with real provider later
            self.pre_arrays = dummy_bt.arrays
            self.report_path = report_path

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function — suggest params and evaluate backtest."""
        params_dict: Dict[str, Any] = {}

        # Parameter ranges from strategy Meta
        ranges = self.strategy_class.Params.Meta.ranges
        categoricals = getattr(self.strategy_class.Params.Meta, "categoricals", {})
        int_params = getattr(self.strategy_class.Params.Meta, "int_params", [])

        for param, (low, high) in ranges.items():
            if param in int_params:
                params_dict[param] = trial.suggest_int(param, int(low), int(high))
            else:
                params_dict[param] = trial.suggest_float(param, low, high)

        for param, choices in categoricals.items():
            params_dict[param] = trial.suggest_categorical(param, choices)

        strategy_params = self.strategy_class.Params(**params_dict)
        strategy = self.strategy_class(
            name=f"{self.strategy_class.__name__}_trial{trial.number}",
            params=strategy_params,
        )

        bt = Backtester(
            timeframes=self.config["timeframes"],
            asset_list=self.config["assets"],
            strategies=[strategy],
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
            initial_capital=self.config.get("initial_capital", 150_000.0),
            leverage=self.config.get("leverage_cap"),
        )

        if self.pre_load:
            bt.arrays = self.pre_arrays.copy()
            bt.pointers = {key: 0 for key in bt.arrays}

        results: BacktestResults = bt.run()

        # Low-sample penalty
        if results.overall_metrics["total_trades"] < self.config.get(
            "min_trades_for_optimization", 30
        ):
            return -1e10

        # Objective selection
        obj_type = self.config.get("objective_metric", "calmar_sharpe")

        if obj_type == "pnl_dd_r2":
            net_pnl = results.overall_metrics["net_pnl"]
            max_dd = abs(results.max_drawdown_pct) + 1e-6
            primary = net_pnl / max_dd
            if len(results.equity_history) > 10:
                times = np.arange(len(results.equity_history))
                _, _, r, _, _ = linregress(
                    times, [e for _, e in results.equity_history]
                )
                r2 = r**2
            else:
                r2 = 0.0
            score = primary * r2

        elif obj_type == "calmar_sharpe":
            calmar = max(results.calmar_ratio, 1e-6)
            score = calmar * (results.sharpe_ratio**2)

        elif obj_type == "sqn":
            score = (
                results.sqn if not np.isnan(results.sqn) and results.sqn > 0 else -1e6
            )

        elif obj_type == "calmar_only":
            score = max(results.calmar_ratio, 1e-6)

        else:
            raise ValueError(f"Unknown objective_metric: {obj_type}")

        trial.report(score, step=1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return score

    def optimize(self) -> optuna.Study:
        """Run the optimization study and generate report for best trial."""
        study_name = self.config.get(
            "study_name", f"{self.strategy_class.__name__}_{dt.date.today():%Y%m%d}"
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            study_name=study_name,
            storage=self.config.get("storage", None),  # None = in-memory
            load_if_exists=True,
        )

        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)

        # Best trial backtest + report
        best_params = self.strategy_class.Params(**study.best_trial.params)
        best_strategy = self.strategy_class(name="Best_Optimized", params=best_params)

        best_bt = Backtester(
            timeframes=self.config["timeframes"],
            asset_list=self.config["assets"],
            strategies=[best_strategy],
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
            initial_capital=self.config.get("initial_capital", 150_000.0),
            leverage=self.config.get("leverage_cap"),
        )

        if self.pre_load:
            best_bt.arrays = self.pre_arrays.copy()
            best_bt.pointers = {key: 0 for key in best_bt.arrays}

        best_results = best_bt.run()
        Reporter(best_results, best_bt).generate_report(
            format="html",
            output_path=self.report_path,
        )

        return study
