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


# normalize multivariate time series
def normalize_time_series(ts, flatten=False):
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


# load challenge data sets
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


# load raw challenge data frame
def load_raw_has_datasets(data_path="../datasets/has2023_master.csv.zip"):
    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    return pd.read_csv(data_path, converters=converters, compression="zip")
