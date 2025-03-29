import matplotlib.pyplot as plt

# Internal state to manage subplot stacking
_subplot_index = 0
_total_subplots = 0
_fig = None


def init_plot_layout(num_subplots=1, figsize=(12, 4 * 1.5)):
    global _subplot_index, _fig, _total_subplots
    _subplot_index = 0
    _total_subplots = num_subplots
    _fig = plt.figure(figsize=figsize)


def highlight_phases(ax, steps, phases, green_ns, green_ew):
    for i in range(len(steps)):
        if phases[i] in green_ns:
            ax.axvspan(steps[i] - 0.5, steps[i] + 0.5, color="lightblue", alpha=0.2)
        elif phases[i] in green_ew:
            ax.axvspan(steps[i] - 0.5, steps[i] + 0.5, color="lightgreen", alpha=0.2)


def plot_lane_metrics(steps, metrics, phases, title, ylabel, green_ns, green_ew):
    global _subplot_index, _fig
    _subplot_index += 1
    ax = _fig.add_subplot(_total_subplots, 1, _subplot_index)

    for label, values in metrics.items():
        ax.plot(steps, values, label=label)

    highlight_phases(ax, steps, phases, green_ns, green_ew)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Simulation Step")
    ax.legend()
    ax.grid(True)


def show_plots():
    plt.tight_layout()
    plt.show()
