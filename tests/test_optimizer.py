"""Tests for Optimizer class covering objectives, pruning, pre-loading, and best-trial rerun."""
import os

import optuna
import pytest

from src.backtester.optimization.optimizer import Optimizer
from src.backtester.strategies.sma_crossover import SMACrossoverStrategy


@pytest.fixture
def optim_config():
    return {
        "assets": ["BTCUSDT"],
        "timeframes": ["1h"],
        "start_date": "2023-01-01",
        "end_date": "2023-01-10",
        "initial_capital": 100_000.0,
        "leverage_cap": None,
        "n_trials": 5,
        "pre_load_data": True,
        "objective_metric": "calmar_sharpe",
        "min_trades_for_optimization": 2,
    }


class TestOptimizer:
    @pytest.mark.parametrize("objective", ["calmar_sharpe", "sqn", "calmar_only"])
    def test_different_objectives(self, optim_config, objective, mocker):
        optim_config["objective_metric"] = objective

        # Mock Backtester.run to return controlled results
        mock_results = mocker.MagicMock()
        mock_results.calmar_ratio = 2.0
        mock_results.sharpe_ratio = 1.5
        mock_results.overall_metrics = {"total_trades": 10}
        mocker.patch("src.backtester.core.backtester.Backtester.run", return_value=mock_results)

        optimizer = Optimizer(optim_config, SMACrossoverStrategy)
        study = optimizer.optimize()

        assert len(study.trials) == 5
        assert study.best_value > 0

    def test_pruning_activates(self, optim_config, mocker):
        # Force early bad trials
        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_res = mocker.MagicMock()
            mock_res.calmar_ratio = 0.1 if call_count < 3 else 3.0
            return mock_res

        mocker.patch("src.backtester.core.backtester.Backtester.run", mock_run)

        optimizer = Optimizer(optim_config, SMACrossoverStrategy)
        study = optimizer.optimize()

        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        assert len(pruned) > 0

    def test_pre_load_data_used(self, optim_config, mocker):
        spy = mocker.spy(SMACrossoverStrategy, "__init__")

        optimizer = Optimizer(optim_config, SMACrossoverStrategy)
        optimizer.optimize()

        # Dummy preload strategy should have been instantiated once
        assert spy.call_count >= 1

    def test_best_trial_rerun_and_report(self, optim_config, mocker, tmp_path):
        mocker.patch("src.backtester.core.backtester.Backtester.run")
        optim_config["report_path"] = str(tmp_path / "best.html")

        optimizer = Optimizer(optim_config, SMACrossoverStrategy)
        optimizer.report_path = optim_config["report_path"]
        optimizer.optimize()

        assert os.path.exists(optim_config["report_path"])