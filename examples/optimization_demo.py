"""
Optimization Demo Script

Demonstrates end-to-end hyperparameter optimization using the built-in Optuna integration.

Steps covered:
1. Define the strategy class with its parameter search space
2. Configure the Optimizer (trials, objective metric, data preloading for speed)
3. Run the Optuna study
4. Extract the best parameters
5. Run a full backtest with the best parameters and generate an HTML report
6. Display live plots: optimization history, parameter importance, and contour plots

Run this script directly to see the full optimization workflow in action.
Ideal for showcasing SEDA's performance-aware optimization capabilities.
"""