import os
import sys

sys.path.insert(0, "../")

from mclasp.segmentation import MultivariateClaSPSegmentation
from benchmark.utils import evalute_segmentation_algorithm, evaluate_candidate
from claspy.segmentation import BinaryClaSPSegmentation
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

import numpy as np

np.random.seed(1379)


def evaluate_pca(dataset, w, cps_true, labels, ts, **seg_kwargs):
    pca = PCA(n_components=1, random_state=2357)
    ts = pca.fit_transform(ts).flatten()

    clasp = BinaryClaSPSegmentation(n_jobs=4)
    cps_pred = clasp.fit_predict(ts)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)


def evaluate_ica(dataset, w, cps_true, labels, ts, **seg_kwargs):
    ica = FastICA(n_components=1, random_state=2357)
    ts = ica.fit_transform(ts).flatten()

    clasp = BinaryClaSPSegmentation(n_jobs=4)
    cps_pred = clasp.fit_predict(ts)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)


def evaluate_rp(dataset, w, cps_true, labels, ts, **seg_kwargs):
    rp = GaussianRandomProjection(n_components=1, random_state=2357)
    ts = rp.fit_transform(ts).flatten()

    clasp = BinaryClaSPSegmentation(n_jobs=4)
    cps_pred = clasp.fit_predict(ts)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)


def evaluate_dist_average(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = MultivariateClaSPSegmentation(aggregation="dist", n_jobs=4)
    cps_pred = clasp.fit_predict(ts)
    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)

def evaluate_dist_average_min(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = MultivariateClaSPSegmentation(aggregation="dist_min", n_jobs=4)
    cps_pred = clasp.fit_predict(ts)
    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)

def evaluate_dist_average_mSTAMP(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = MultivariateClaSPSegmentation(aggregation="dist_mSTAMP", n_jobs=4)
    cps_pred = clasp.fit_predict(ts)
    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)

def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    competitors = [
        # ("PCA", evaluate_pca),
        # ("ICA", evaluate_ica),
        # ("RP", evaluate_rp),
        # ("DistAvg", evaluate_dist_average),
        ("mSTAMP", evaluate_dist_average_mSTAMP),
        ("MinDistAvg", evaluate_dist_average_min),
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
    exp_path = "../experiments/ablation_study/"
    n_jobs, verbose = 32, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    for bench in ("HAS_train",):
        evaluate_competitor(bench, exp_path, n_jobs, verbose)
