import numpy as np
from itertools import product
import time
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Core regret / utility functions
# ---------------------------------------------------------------------------

def regret_given_cost_vectorized_fixed(y, lr_predictions, costs=0, verbose=False):
    """Compute vectorized regret with cost adjustment using argmax (maximization).

    Args:
        y: Array of true utility values, shape (n_samples, n_models).
        lr_predictions: Array of predicted utility values, shape (n_samples, n_models).
        costs: Scalar or array of costs subtracted from both y and predictions.
        verbose: If True, print per-model regret statistics.

    Returns:
        List of mean regrets for each method followed by the mean regret of the
        predicted selection.
    """
    y = y - costs
    lr_predictions = lr_predictions - costs
    optimal_methods = np.argmax(y, axis=1)
    predicted_methods = np.argmax(lr_predictions, axis=1)
    optimal_values = y[np.arange(len(y)), optimal_methods]

    num_methods = y.shape[1]
    regret_methods_mean = []
    regret_methods_stderr = []
    for i in range(num_methods):
        regret_method = optimal_values - y[:, i]
        regret_methods_mean.append(regret_method.mean())
        regret_methods_stderr.append(regret_method.std() / np.sqrt(len(regret_method)))

    predicted_values = y[np.arange(len(y)), predicted_methods]
    regrets_mine = optimal_values - predicted_values
    correct_predictions = np.sum(predicted_methods == optimal_methods)

    if verbose:
        for i in range(num_methods):
            print(f'{np.mean(optimal_methods == i):.4f}', regret_methods_mean[i], '+-', regret_methods_stderr[i])
        print(f'{correct_predictions / len(lr_predictions):.4f}',
              regrets_mine.mean(), '+-',
              regrets_mine.std() / np.sqrt(len(regrets_mine)))

    return regret_methods_mean + [regrets_mine.mean()]


def utility_given_cost_vectorized_fixed(y, lr_predictions, costs=0, scaling_factor=0, verbose=False):
    """Compute mean utility and cost for each method and for predicted selections.

    Args:
        y: Array of true utility values, shape (n_samples, n_models).
        lr_predictions: Array of predicted utility values, shape (n_samples, n_models).
        costs: Scalar or array of per-model costs.
        scaling_factor: Factor applied to costs before subtracting from predictions.
        verbose: If True, print detailed output.

    Returns:
        Tuple of (utility_means, cost_means) where each list contains one entry
        per method followed by the predicted-selection entry.
    """
    lr_predictions = lr_predictions - costs * scaling_factor
    predicted_methods = np.argmax(lr_predictions, axis=1)

    num_methods = y.shape[1]
    utility_methods_mean = []
    utility_methods_stderr = []
    cost_methods_mean = []
    for i in range(num_methods):
        utility_method = y[:, i]
        utility_methods_mean.append(utility_method.mean())
        utility_methods_stderr.append(utility_method.std() / np.sqrt(len(utility_method)))
        cost_methods_mean.append(costs[i])

    predicted_values = y[np.arange(len(y)), predicted_methods]
    utilities_mine = predicted_values
    predicted_costs = costs[predicted_methods]

    return utility_methods_mean + [utilities_mine.mean()], cost_methods_mean + [predicted_costs.mean()]


# ---------------------------------------------------------------------------
# Helpers for cost-grid experiments
# ---------------------------------------------------------------------------

def _build_percentile_cost_ranges(y_combined, num_methods, percentile_lo=25,
                                  percentile_hi=75, n_points=10,
                                  spread_scale=11/10, exclude_last=1):
    """Build per-method cost linspaces from percentile-based spread.

    Args:
        y_combined: Utility matrix for computing percentiles.
        num_methods: Number of models to generate ranges for.
        percentile_lo: Lower percentile for spread calculation.
        percentile_hi: Upper percentile for spread calculation.
        n_points: Number of points per cost range.
        spread_scale: Scale factor over range to be inlcusive at end points.
        exclude_last: Number of trailing columns to exclude from spread calc.

    Returns:
        List of per-method cost linspaces.
    """
    lo = np.percentile(y_combined, percentile_lo, axis=0)
    hi = np.percentile(y_combined, percentile_hi, axis=0)
    spread = hi - lo
    min_spread = np.min(spread[:-exclude_last], axis=0)
    return [np.linspace(hi[i] - min_spread * spread_scale, hi[i], n_points)
            for i in range(num_methods)]


def _parallel_regret_over_costs(y, predictions, cost_combinations, n_jobs=8):
    """Run regret_given_cost_vectorized_fixed in parallel over cost combinations.

    Args:
        y: Ground truth utility matrix.
        predictions: Predicted scores.
        cost_combinations: List of cost vectors to evaluate.
        n_jobs: Number of parallel jobs.

    Returns:
        Mean regret vector averaged over all cost combinations.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(regret_given_cost_vectorized_fixed)(y, predictions, costs, verbose=False)
        for costs in cost_combinations
    )
    return np.sum(results, axis=0) / len(cost_combinations)


# ---------------------------------------------------------------------------
# Experiment functions
# ---------------------------------------------------------------------------

LATENCY_COSTS = np.array([30, 10, 10, 0.5, 289])
MEMORY_COSTS = np.array([6, 24, 16, 6])
LATENCY_MEMORY_COSTS = LATENCY_COSTS[:-1] * MEMORY_COSTS

def latency_memory_exps(utility_y_test, utility_test_predictions, utility_y_train_val, utility_train_val_predictions, utility_y_test_with_pass, utility_test_predictions_with_pass, utility_y_train_val_with_pass, utility_train_val_predictions_with_pass, verbose=False):
    """Run latency, memory, and combined latency-memory cost-scaling experiments.

    Returns:
        Tuple of (utilities_latency, utilities_memory, utilities_latency_memory) arrays.
    """
    latency_scaling = np.linspace(0, .3, 10000)
    utilities_latency = np.array(Parallel(n_jobs=8)(
        delayed(utility_given_cost_vectorized_fixed)(
            utility_y_test_with_pass, utility_test_predictions_with_pass, LATENCY_COSTS, i,
            verbose=False
        ) for i in latency_scaling
    ))

    memory_scaling = np.linspace(0, .25, 10000)
    utilities_memory = np.array(Parallel(n_jobs=8)(
        delayed(utility_given_cost_vectorized_fixed)(
            utility_y_test, utility_test_predictions, MEMORY_COSTS, i,
            verbose=False
        ) for i in memory_scaling
    ))

    latency_memory_scaling = np.linspace(0, .025, 10000)
    utilities_latency_memory = np.array(Parallel(n_jobs=8)(
        delayed(utility_given_cost_vectorized_fixed)(
            utility_y_test, utility_test_predictions, LATENCY_MEMORY_COSTS, i,
            verbose=False
        ) for i in latency_memory_scaling
    ))

    return utilities_latency, utilities_memory, utilities_latency_memory


def single_point_exp(utility_y_test, utility_test_predictions, utility_y_train_val, utility_train_val_predictions, utility_y_test_with_pass, utility_test_predictions_with_pass, utility_y_train_val_with_pass, utility_train_val_predictions_with_pass, verbose=False):
    """Run a single-point regret experiment with no cost adjustment."""
    regrets = regret_given_cost_vectorized_fixed(utility_y_test, utility_test_predictions, verbose=True)
    return regrets


def full_cost_range_exp(utility_y_test, utility_test_predictions, utility_y_train_val, utility_train_val_predictions, utility_y_test_with_pass, utility_test_predictions_with_pass, utility_y_train_val_with_pass, utility_train_val_predictions_with_pass, verbose=False):
    """Run a full cost-range experiment over percentile-based cost combinations with pass."""
    y_combined_with_pass = np.concatenate([utility_y_test_with_pass, utility_y_train_val_with_pass], axis=0)
    cost_ranges = _build_percentile_cost_ranges(y_combined_with_pass, num_methods=5)
    all_costs = list(product(*cost_ranges))
    return _parallel_regret_over_costs(
        utility_y_test_with_pass, utility_test_predictions_with_pass, all_costs
    )


def gen_cost_range_exp(utility_y_test, utility_test_predictions, utility_y_train_val, utility_train_val_predictions, utility_y_test_with_pass, utility_test_predictions_with_pass, utility_y_train_val_with_pass, utility_train_val_predictions_with_pass, verbose=False):
    """Run a full cost-range experiment with only viewpoint depedent models over percentile-based cost combinations."""
    y_combined = np.concatenate([utility_y_test, utility_y_train_val], axis=0)
    cost_ranges = _build_percentile_cost_ranges(
        y_combined, num_methods=4, n_points=20, spread_scale=21/20
    )
    all_costs = list(product(*cost_ranges))
    start_time = time.time()
    result = _parallel_regret_over_costs(utility_y_test, utility_test_predictions, all_costs)
    if verbose:
        print(f"Exp9 results: {result}")
        print(f"Time for exp8: {time.time() - start_time}")
    return result