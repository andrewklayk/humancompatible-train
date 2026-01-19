import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_log_arrays(logs, subset='train', stats_to_extract=('mean', 'std')):
    """
    Extracts and processes loss and constraint data from algorithm logs.

    Args:
        logs: A list of lists of histories returned by algorithms, with shape (n_algs, n_runs).
        subset: The subset of data to extract (e.g., 'train' or 'val'). Defaults to 'train'.
        stats_to_extract: The statistics to extract from the data. Gets applied to dataframes as an aggregation function.

    Returns:
        A tuple of tuples: ((losses, losses_mean, losses_std), (constraints, constraints_mean, constraints_std))
    """
    get_constr_cols = lambda cols: [col for col in cols if col.startswith('c_') and col.endswith(subset)]
    # Convert each run's log to a DataFrame and set 'epoch' as the index
    logs_pd = [ [ pd.DataFrame(history).set_index('epoch')
                 for history in runs_log ]
                 for runs_log in logs ]
    # Extract losses for the specified subset
    losses = [ [ run[f'{subset}_loss']
                for run in runs_log ]
                for runs_log in logs_pd ]
    # Extract constraints for the specified subset
    constraints = [
        [ run [ get_constr_cols(run.columns) ]
          for run in runs_log ] 
          for runs_log in logs_pd ]

    # Group all runs by epoch and compute statistics
    grouped = [ pd.concat(runs_log).groupby(level=0) for runs_log in logs_pd ]

    loss_stats = []
    constraints_stats = []
    for stat in stats_to_extract:
        # Compute statistic for each algorithm's logs
        stat_dfs = [grouped_alg_log.agg(stat) for grouped_alg_log in grouped]
        loss_stat = [ stat_df[f'{subset}_loss'].to_numpy() for stat_df in stat_dfs ]
        constraints_stat = [
            stat_df[
                get_constr_cols(stat_df.columns)
            ].to_numpy().T
            for stat_df in stat_dfs ]
        loss_stats.append(loss_stat)
        constraints_stats.append(constraints_stat)

    return (
        (losses, *loss_stats),
        (constraints, *constraints_stats)
    )




def plot_mean_std(loss_mean, loss_std, constraint_mean, constraint_std, figure = None, bounds = None, titles=None, color=None, subset = ''):
    """
    Plot each array of values with standard deviation bounds.

    Parameters:
    - data_arrays: List of arrays, each containing the data points.
    - std_arrays: List of arrays, each containing the standard deviations for the data points.
    - titles: List of titles for each plot. If None, no titles are used.
    """

    if len(loss_mean) != len(loss_std):
        raise ValueError("The number of data arrays and standard deviation arrays must be the same.")

    num_plots = len(loss_mean)
    n_constraints = len(constraint_mean[0])
    
    if figure:
        f, axes = figure
    else:
        f, axes = plt.subplots(nrows = 1 + n_constraints, ncols = num_plots, figsize=(6 * num_plots, 6 * n_constraints),sharex='col', sharey='row')
        axes = axes.T

    if num_plots == 1:
        axes = [axes]

    for i, (l_m, l_std, c_m, c_std, ax_col) in enumerate(zip(loss_mean, loss_std, constraint_mean, constraint_std, axes)):
        x = np.arange(len(l_m))
        ax = ax_col[0]
        ax.plot(x, l_m, label='Mean' + ' ' + subset, color=color)
        ax.scatter(x, l_m, color=color, s=20)
        ax.fill_between(x, l_m - l_std, l_m + l_std, color=color, alpha=0.2, label='±1 std')
        ax.grid()
        ax.legend()
        ax.set_ylabel('Loss')
        if titles and i < len(titles):
            ax.set_title(titles[i])
        
        for j, (c_j_mean, c_j_std, bound_j) in enumerate(zip(c_m, c_std, bounds), start = 1):
            ax = ax_col[j]
            ax.set_ylabel(f'Constraint {j}')
            ax.plot(x, c_j_mean, label='Mean' + ' ' + subset, color=color)
            ax.scatter(x, c_j_mean, color=color, s=20)
            ax.fill_between(x, c_j_mean - c_j_std, c_j_mean + c_j_std, color=color, alpha=0.2, label='±1 std')
            ax.hlines(bound_j, 0, x.max(), ls='--', color='black')
            ax.legend()

    plt.tight_layout()
    return f, axes