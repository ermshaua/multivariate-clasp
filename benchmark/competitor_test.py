import os
import sys

from benchmark.utils import evalute_segmentation_algorithm

sys.path.insert(0, "../")

from mclasp.segmentation import MultivariateClaSPSegmentation

import daproli as dp
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import f_measure, covering
from claspy.segmentation import BinaryClaSPSegmentation
from src.utils import load_has_datasets, load_datasets

import numpy as np

np.random.seed(1379)


def evaluate_clasp(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = BinaryClaSPSegmentation(n_jobs=4) #
    cps_pred = clasp.fit_predict(ts[:, 1]) #

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)


def evaluate_mclasp(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = MultivariateClaSPSegmentation(n_jobs=4, threshold=1e-30) # n_segments=len(cps_true)+1,
    cps_pred = clasp.fit_predict(ts)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)


def evaluate_candidate(dataset_name, candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name == "HAS":
        df = load_has_datasets()
    else:
        df = load_datasets(dataset_name)

    # apply selection
    # df = df.sample(n=25, random_state=2357)

    df_cand = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df.iterrows()), disable=verbose < 1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["dataset", "true_cps", "found_cps", "f1_score", "covering_score"]

    df_cand = pd.DataFrame.from_records(
        df_cand,
        index="dataset",
        columns=columns,
    )

    print(
        f"{dataset_name} {candidate_name}: mean_f1_score={np.round(df_cand.f1_score.mean(), 3)}, mean_covering_score={np.round(df_cand.covering_score.mean(), 3)}")
    return df_cand


def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    competitors = [
        # ("ClaSP", evaluate_clasp),
        ("MClaSP", evaluate_mclasp),
    ]

    for candidate_name, eval_func in competitors:
        print(f"Evaluating competitor: {candidate_name}")

        df = evaluate_candidate(
            dataset_name,
            candidate_name,
            eval_func=eval_func,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        df.to_csv(f"{exp_path}{dataset_name}_{candidate_name}.csv.gz", compression='gzip')


if __name__ == '__main__':
    exp_path = "../experiments/competitor/"
    n_jobs, verbose = 32, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    for bench in ("HAS",):
        evaluate_competitor(bench, exp_path, n_jobs, verbose)