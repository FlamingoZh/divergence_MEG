import os

home=os.path.expanduser("~")

def topoplot(
    mat,
    nrow=4,
    ncol=5,
    time_step=25,
    time_start=0,
    cmap="RdBu_r",
    vmin=-0.1,
    vmax=0.1,
    figsize=(15, 15),
    fontsize=16,
):
    """Creates helmet plots for sensor-space MEG data (based on MNE visualization)

    Args:
        mat (2d numpy array): data to plot of size N sensors x M timepoints
        nrow (int, optional): number of rows in plot. Defaults to 4.
        ncol (int, optional): number of columns in plot. Defaults to 5.
        time_step (int, optional): time window length. Defaults to 25.
        time_start (int, optional): what time to start plotting. Defaults to 0.
        cmap (str, optional): colormap name. Defaults to 'RdBu_r'.
        vmin (float, optional): sets the colorbar min. Defaults to -0.1.
        vmax (float, optional): sets the colorbar max. Defaults to 0.1.
        figsize (tuple, optional): figure size. Defaults to (15,15).
        fontsize (int, optional): font size. Defaults to 16.

    Returns:
        figure handle
    """

    from sklearn.metrics.pairwise import euclidean_distances
    import csv
    import numpy as np
    import mne
    import matplotlib.pyplot as plt

    # gets sensor locations
    with open(f"{home}/Desktop/Sherlock_MEG/HP_data/meg/locations.txt", "r") as f:
        locs = csv.reader(f, delimiter=",")
        loc306 = np.asarray(
            [[float(w1[0].split(" ")[1]), float(w1[0].split(" ")[2])] for w1 in locs]
        )
    loc102 = loc306[::3]
    loc = {306: loc306, 102: loc102}[mat.shape[0]]  # pick the correct channel locations

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
    i = 0
    for row in ax:
        for col in row:
            if i < mat.shape[1]:
                h = mne.viz.plot_topomap(
                    mat[:, i],
                    loc,
                    vlim=(vmin,vmax),
                    axes=col,
                    cmap=cmap,
                    show=False,
                    contours = 0,
                )
            i += 1
    i = 0
    for row in ax:
        for col in row:
            col.set_title(
                "{} to {} ms".format(
                    i * time_step + time_start, (i + 1) * time_step + time_start
                ),
                fontsize=fontsize,
            )
            i += 1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.00, 0.15, 0.01, 0.71])
    cbar = fig.colorbar(h[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize)

    # set the font type and size of the colorbar
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_size(20)

    plt.tight_layout()
    return fig

def single_topoplot(
    mat,
    cmap="RdBu_r",
    vmin=-0.1,
    vmax=0.1,
    figsize=(5,5),
    fontsize=16,
):
    """Creates single helmet plot for sensor-space MEG data (based on MNE visualization)

    Args:
        mat (2d numpy array): data to plot of size N sensors x M timepoints
        cmap (str, optional): colormap name. Defaults to 'RdBu_r'.
        vmin (float, optional): sets the colorbar min. Defaults to -0.1.
        vmax (float, optional): sets the colorbar max. Defaults to 0.1.
        figsize (tuple, optional): figure size. Defaults to (15,15).
        fontsize (int, optional): font size. Defaults to 16.

    Returns:
        figure handle
    """

    from sklearn.metrics.pairwise import euclidean_distances
    import csv
    import numpy as np
    import mne
    import matplotlib.pyplot as plt

    # gets sensor locations
    with open(f"{home}/Desktop/Sherlock_MEG/HP_data/meg/locations.txt", "r") as f:
        locs = csv.reader(f, delimiter=",")
        loc306 = np.asarray(
            [[float(w1[0].split(" ")[1]), float(w1[0].split(" ")[2])] for w1 in locs]
        )
    loc102 = loc306[::3]
    loc = {306: loc306, 102: loc102}[mat.shape[0]]  # pick the correct channel locations

    fig = plt.figure(figsize=figsize)
    h = mne.viz.plot_topomap(
                    mat,
                    loc,
                    vlim=(vmin,vmax),
                    cmap=cmap,
                    show=False,
                    size=5,
                )

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(h[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize)

    # set the font type and size of the colorbar
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_size(20)

    plt.tight_layout()
    return fig
