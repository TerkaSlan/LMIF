import matplotlib.pyplot as plt
import numpy as np
from typing import List
from lmi.utils import load_json, create_dir
import os
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

colors = {
    "LR": (170/255, 0,       120/255),
    "RF": (150/255, 225/255, 0),
    "GMM": (106/255, 90/255, 205/255),
    "NN": (0/255,   0/255,   120/255),
    "KMeans": (0/255,   0/255,   120/255),
    "multilabel-NN": (0/255,   200/255, 195/255),
    "NNMulti": (0/255,   200/255, 195/255),
    "KMeansLogReg": (0/255,   200/255, 195/255),
    "Mtree": (255/255, 205/255, 25/255),
    "Mindex": (255/255, 205/255, 25/255),
    "BayesianGMM": (255/255, 150/255, 50/255)
}

lines = ['-', '--', '-.', (0, (1, 1)), (0, (1, 1)), '-.']
markers = ['o', 's', 'D', '^', 'P', 'o']
line_width = 2


def create_single_plot(
    fig,
    ax,
    x,
    y,
    i,
    x_label,
    y_label,
    line_label,
    x_ticks=None,
    y_ticks=np.arange(0, 1.1, 0.1)
):
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
    ax.plot(
        x,
        y,
        markers[i % 5],
        linestyle=lines[i % 5],
        color=colors[line_label] if line_label in colors else colors['LR'],
        label=line_label,
        linewidth=line_width
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [
                f'{round(x_tick/1000, 1)}k' if i == 0 else f'{int(x_tick/1000)}k' for i, x_tick in enumerate(x_ticks)
            ]
        )

    y_tick_labels = [str(round(v, 2))[1:] if round(v, 2) != 1.0 else round(v, 2) for v in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.legend()
    ax.grid(b=True, which='major', color='k', linestyle='-', linewidth=0.2)
    return fig, ax


def save_figure(fig, filename, dir_to_save_to, experiment_dirs=None):
    if not dir_to_save_to:
        for exp_dir in experiment_dirs:
            fig.savefig(f'{exp_dir}/{filename}.png')
        return f'{exp_dir}/{filename}.png'
    else:
        create_dir(dir_to_save_to)
        fig.savefig(f'{dir_to_save_to}/{filename}.png')
        return f'{dir_to_save_to}/{filename}.png'


class Plot:

    def __init__(self, experiment_dirs: List[str]):
        self.experiment_dirs = experiment_dirs
        self.experiment_info = []

    def get_experiment_infos(self, exp_names=None):

        self.times = []
        self.scores = []
        self.models = []
        self.stop_conditions = []
        self.stop_conditions_perc = []

        if exp_names and len(exp_names) == len(self.experiment_dirs):
            self.models = exp_names

        for experiment_dir in self.experiment_dirs:
            try:
                summary = load_json(os.path.join(experiment_dir, 'summary.json'))
            except AssertionError:
                print(f'Summary not found for: {experiment_dir}, skipping')
            exp_times = []
            exp_scores = []
            for _, results in summary['results'].items():
                exp_times.append(results['time'])
                exp_scores.append(results['score'])

            self.times.append(exp_times)
            self.scores.append(exp_scores)
            if self.stop_conditions != []:
                assert len(list(summary['results'].keys())) == len(self.stop_conditions), \
                    "Experiments don't have a consistent number of stop conditions."
            else:
                self.stop_conditions = [int(k) for k in list(summary['results'].keys())]
            self.stop_conditions_perc = summary['stop_conditions_perc']

    def plot_experiments(
        self,
        save=False,
        filename='plot',
        dir_to_save_to=None,
        x_ticks=[500, 50_000, 100_000, 200_000, 300_000]
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
        fig, axs = plt.subplots(figsize=(12, 4), ncols=2, nrows=1)
        for i, (model, scores, times) in enumerate(zip(self.models, self.scores, self.times)):

            fig, ax0 = create_single_plot(
                fig,
                axs[0],
                x=self.stop_conditions,
                y=scores,
                i=i,
                x_label='Number of objects in buckets visited',
                y_label='Recall',
                line_label=model,
                x_ticks=x_ticks
            )
            fig, ax1 = create_single_plot(
                fig,
                axs[1],
                x=times,
                y=scores,
                i=i,
                x_label='Time (s)',
                y_label='Recall',
                line_label=model
            )
        if len(self.models) != 0:
            if save:
                return save_figure(fig, f'{filename}time-recall-stopcond-recall', dir_to_save_to, self.experiment_dirs)
            else:
                fig.show()


def plot_boxplots(
    data_a,
    data_b,
    ticks,
    save=False,
    filename='',
    dir_to_save_to='',
    labels=['', ''],
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
    def set_box_color(bp, color):
        for i in ['boxes', 'medians', 'caps', 'whiskers']:
            plt.setp(bp[i], color=color)

    fig = plt.figure(figsize=(5, 3))

    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0 - 0.3, sym='', widths=0.5)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0 + 0.3, sym='', widths=0.5)
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')

    plt.plot([], c='#D7191C', label=labels[0])
    plt.plot([], c='#2C7BB6', label=labels[1])
    plt.legend(prop={'size': 8}, loc=legend_loc)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(-0.01, 1.05)
    plt.grid(axis='y')
    if save:
        return save_figure(fig, f'{filename}boxplots', dir_to_save_to)
    else:
        fig.show()
