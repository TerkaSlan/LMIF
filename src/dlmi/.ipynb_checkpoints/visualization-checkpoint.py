import matplotlib.pyplot as plt
import numpy as np


def plot_knn_distribution_in_leaf_nodes(leaf_nodes_dist, query_id, found_among_nns, filename):

    leaf_nodes = list(leaf_nodes_dist.index)
    group_found = leaf_nodes_dist[leaf_nodes_dist.index.isin(found_among_nns)]
    group_found_idx = [leaf_nodes.index(leaf_node) for leaf_node in group_found.index]
    group_not_found = leaf_nodes_dist[~leaf_nodes_dist.index.isin(found_among_nns)]
    group_not_found_idx = [leaf_nodes.index(leaf_node) for leaf_node in group_not_found.index]

    fig = plt.figure(figsize=(20, 7))
    plt.title(
        f'Distribution of 100-NN ground truth in the buckets of DLMI for query: {query_id}'
    )
    for group, ticks, label in zip(
        [group_found, group_not_found],
        [group_found_idx, group_not_found_idx],
        ['Found', 'Not found']
    ):
        keys = group.keys()
        values = group.values

        plt.bar(ticks, values, label=label)
        plt.grid(True)
        plt.xticks(ticks, keys, rotation=45)

    plt.legend()
    plt.xticks([i for i in range(len(leaf_nodes))], leaf_nodes, rotation=45)
    if filename is not None:
        fig.savefig(filename)


def plot_build(info_df, filename, step):
    x = info_df[info_df['op'] == 'REORGANIZATION']['#-objects']
    x = [str(x_)[:-3] + 'k' if str(x_).endswith('000') else x_ for x_ in x]
    y1 = info_df[info_df['op'] == 'REORGANIZATION']['time-taken'].cumsum()
    y2 = info_df[info_df['op'] == 'REORGANIZATION']['size']

    fig, axs = plt.subplots(figsize=(20, 10), ncols=1, nrows=2)
    fig, ax0 = create_single_plot(
        fig,
        axs[0],
        x=x,
        y=y1,
        i=0,
        x_label='#-objects',
        y_label='(Cumulative) Build time (s)',
        x_ticks = [100, 1_000, 10_000, 100_000, 1_000_000]
    )
    fig, ax1 = create_single_plot(
        fig,
        axs[1],
        x=x,
        y=y2,
        i=0,
        x_label='#-objects',
        y_label='Size (MB)',
        x_ticks = [100, 1_000, 10_000, 100_000, 1_000_000]
    )
    fig.suptitle(f'Index build time and size -- step {step}')
    if filename is not None:
        fig.savefig(filename)


def plot_search(search_df, n_leaf_nodes, filename, step, search_type='leaf'):
    recalls = []
    mean_recalls = []
    stop_conds = []
    n_queries = 0

    for n, g in search_df.groupby(['sc']):
        n_queries = g['recall'].values.shape[0]
        if search_type == 'leaf':
            stop_conds.append(f"{n} / {n_leaf_nodes}\n{round((n/n_leaf_nodes)*100, 2)}%\n{round(g['time'].mean(),4)} s")
        elif search_type == 'time':
            stop_conds.append(f"{n}")
        recalls.append(g['recall'].values)
        mean_recalls.append(np.mean(g['recall'].values))

    fig = plt.figure(figsize=(10, 7))
    condition = 'time' if search_type == 'time' else '# leaf nodes in candidate answer'
    fig.suptitle(f'Recall of search given {condition} for index building step {step}. {n_queries} queries')
    fig = plot_line_with_boxplots(
        fig,
        recalls,
        mean_recalls,
        stop_conds,
        x_label='# leaf nodes in candidate answer \n time taken (s)' if search_type == 'leaf' else 'time (s)'
    )
    fig.savefig(filename)


def create_single_plot(
    fig,
    ax,
    x,
    y,
    i,
    x_label,
    y_label,
    label='',
    plot_type='plot',
    title=''
):
    lines = ['-', '--', '-.', (0, (1, 1)), (0, (1, 1)), '-.']
    markers = ['o', 's', 'D', '^', 'P', 'o']
    line_width = 2

    """ Creates one line in the plot of connected scatterpoints with `x` and `y`
    Parameters
    -------
    fig : plot figure
    ax : plot axis
    x : List[float] or List[int]
        Contents for the x-axis. Time taken or stop-conditions met.
    y : List[float]
        Contents for the y-axis. Recall.
    i : int
        Current line counter
    Returns
    -------
    fig, ax
    """
    if plot_type == 'plot':
        p0, = ax.plot(
            x,
            y,
            markers[i % 5],
            linestyle=lines[i % 5],
            linewidth=line_width,
            label=label
        )
    elif plot_type == 'bar':
        p0 = ax.bar(
            x,
            y,
            width=20
        )
        ax.set_yscale('log')
    else:
        p0 = ax.scatter(
            x,
            y
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    #ax.legend()
    ax.grid(b=True, which='major', color='gray', linestyle='-', linewidth=0.2)
    return fig, ax, p0


def plot_line_with_boxplots(
    fig,
    data_values,
    mean_data_values,
    ticks,
    x_label='',
    legend_loc='lower right'
):
    """ Creates time-recall and stopcond-recall plots of connected scatterpoints.

    Parameters
    -------
    save : bool
        Should the plot be saved.
    filename : str
        Filename to save as.
    dir_to_save_to : str
        Directory to save to.
    """

    for pos, data_value in enumerate(data_values):
        plt.boxplot(data_value, positions=[pos], sym='', widths=0.5)

    plt.plot([i for i in range(len(data_values))], mean_data_values, marker='o')
    plt.legend(prop={'size': 8}, loc=legend_loc)
    plt.xticks(range(0, len(ticks), 1), ticks)
    plt.ylim(-0.01, 1.05)
    plt.grid(axis='y')
    plt.xlabel(x_label)
    plt.ylabel('Recall')

    return fig
