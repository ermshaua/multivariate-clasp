import os
import shutil
import stat
import tempfile
import subprocess

from numba import njit
import matplotlib.pyplot as plt

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd


def normalize_time_series(ts):
    flatten = False

    if ts.ndim == 1:
        ts = ts.reshape(-1,1)
        flatten = True

    for dim in range(ts.shape[1]):
        channel = ts[:,dim]

        # min-max normalize channel
        try:
            channel = np.true_divide(channel - channel.min(), channel.max() - channel.min())
        except FloatingPointError:
            pass

        # interpolate (if missing values are present)
        channel[np.isinf(channel)] = np.nan
        channel = pd.Series(channel).interpolate(limit_direction="both").to_numpy()

        # there are series that still contain NaN values
        channel[np.isnan(channel)] = 0

        ts[:,dim] = channel

    if flatten:
        ts = ts.flatten()

    return ts


def load_datasets(dataset, selection=None, normalize=True):
    desc_filename = ABS_PATH + f"/../datasets/{dataset}/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    prop_filename = ABS_PATH + f"/../datasets/{dataset}/properties.txt"
    prop_file = []

    with open(prop_filename, 'r') as file:
        for line in file.readlines(): prop_file.append(line.split(","))

    assert len(desc_file) == len(prop_file), "Description and property file have different records."

    df = []

    for idx, (desc_row, prop_row) in enumerate(zip(desc_file, prop_file)):
        if selection is not None and idx not in selection: continue
        assert desc_row[0] == prop_row[0], f"Description and property row {idx} have different records."

        (ts_name, window_size), change_points = desc_row[:2], desc_row[2:]
        labels = prop_row[1:]

        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        path = ABS_PATH + f'/../datasets/{dataset}/'

        if os.path.exists(path + ts_name + ".txt"):
            ts = np.loadtxt(fname=path + ts_name + ".txt", dtype=np.float64)
        else:
            ts = np.load(file=path + "data.npz")[ts_name]

        # min-max normalize ts
        if normalize: ts = normalize_time_series(ts)

        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]),
                   np.array([int(_) for _ in labels]), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "window_size", "change_points", "labels", "time_series"])


def load_has_datasets(selection=None, normalize=True, split=None):
    data_path = ABS_PATH + "/../datasets/has2023_master.csv.zip"

    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]

    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}

    df_has = pd.read_csv(data_path, converters=converters, compression="zip")

    df = []
    sample_rate = 50

    for _, row in df_has.iterrows():
        if selection is not None and row.ts_challenge_id not in selection: continue
        if split is not None and row.split != split: continue

        ts_name = f"{row.group}_subject{row.subject}_routine{row.routine} (id{row.ts_challenge_id})"

        label_mapping = {label: idx for idx, label in enumerate(np.unique(row.activities))}
        labels = np.array([label_mapping[label] for label in row.activities])

        if row.group == "indoor":
            ts = np.hstack((
                row["x-acc"].reshape(-1, 1),
                row["y-acc"].reshape(-1, 1),
                row["z-acc"].reshape(-1, 1),
                row["x-gyro"].reshape(-1, 1),
                row["y-gyro"].reshape(-1, 1),
                row["z-gyro"].reshape(-1, 1),
                row["x-mag"].reshape(-1, 1),
                row["y-mag"].reshape(-1, 1),
                row["z-mag"].reshape(-1, 1)
            ))
        elif row.group == "outdoor":
            ts = np.hstack((
                row["x-acc"].reshape(-1, 1),
                row["y-acc"].reshape(-1, 1),
                row["z-acc"].reshape(-1, 1),
                row["x-mag"].reshape(-1, 1),
                row["y-mag"].reshape(-1, 1),
                row["z-mag"].reshape(-1, 1),
                # leads to errors
                # row["lat"].reshape(-1, 1),
                # row["lon"].reshape(-1, 1),
                # row["speed"].reshape(-1, 1)
            ))
        else:
            raise ValueError("Unknown group in HAS dataset.")

        # min-max normalize ts
        if normalize: ts = normalize_time_series(ts)

        df.append((ts_name, sample_rate, row.change_points, labels, ts))

    if selection is None:
        selection = np.arange(len(df))

    return pd.DataFrame.from_records(
        df,
        columns=["dataset", "window_size", "change_points", "labels", "time_series"],
    ).iloc[selection, :]


def load_raw_has_datasets(data_path="../datasets/has2023_master.csv.zip"):
    """
    Load the given CSV file containing the sensor data for the challenge.
    Returns a pandas DataFrame where each column is a sensor measurement and
    each row corresponds to a single time series of sensor data.

    Parameters
    ----------
    data_path : str, default: "../datasets/has2023.csv.zip".
        Path to the csv file to be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sensor data for the challenge.

    Examples
    --------
    >>> data = load_raw_has_datasets()
    >>> data.head()
    """
    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    return pd.read_csv(data_path, converters=converters, compression="zip")


def visualize_activity_data(title, sensor_names, time_series, change_points, activities, show=True, file_path=None, sample_rate=50,
                       font_size=18):
    """
    Plots multivariate time series data segmented by change points and colored by activity labels.

    Parameters
    ----------
    title : str
        The title of the time series plot.
    sensor_names : list of str
        List of sensor names corresponding to each time series.
    time_series : list of np.ndarray
        List of arrays containing the time series data from sensors.
    change_points : np.ndarray
        Array of change points indicating the indices where activity changes.
    activities : list of str
        List of activity labels corresponding to the segments between change points.
    show : bool, optional
        Whether to display the plot. Default is True.
    file_path : str, optional
        If provided, saves the plot to the given file path.
    sample_rate : int, optional
        Sample rate of the time series data. Default is 50.
    font_size : int, optional
        Font size for the axis labels and title. Default is 18.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The last Axes object of the plot.
    """
    plt.clf()
    fig, axes = plt.subplots(
        len(time_series),
        sharex=True,
        gridspec_kw={'hspace': .15},
        figsize=(20, len(time_series) * 2)
    )

    activity_colours = {}
    idx = 0

    for activity in activities:
        if activity not in activity_colours:
            activity_colours[activity] = f"C{idx}"
            idx += 1

    for ts, sensor, ax in zip(time_series, sensor_names, axes):
        if len(ts) > 0:
            segments = [0] + change_points.tolist() + [ts.shape[0]]
            for idx in np.arange(0, len(segments) - 1):
                ax.plot(
                    np.arange(segments[idx], segments[idx + 1]),
                    ts[segments[idx]:segments[idx + 1]],
                    c=activity_colours[activities[idx]]
                )

        ax.set_ylabel(sensor, fontsize=font_size)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

    axes[0].set_title(title, fontsize=font_size)
    axes[-1].set_xticklabels([f"{int(tick / sample_rate)}s" for tick in axes[-1].get_xticks()])

    if show is True:
        plt.show()

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")

    return ax