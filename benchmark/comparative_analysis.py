import os
import sys

sys.path.insert(0, "../")

from benchmark.utils import evaluate_candidate
from benchmark.variants import evaluate_pca, evaluate_dist_average, \
    evaluate_profile_average

import numpy as np

np.random.seed(1379)


def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    competitors = [
        ("PCA", evaluate_pca),
        ("DistAvg", evaluate_dist_average),
        ("ProfileAvg", evaluate_profile_average),
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
    exp_path = "../experiments/competitors/"
    n_jobs, verbose = 32, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    for bench in ("HAS_test",):
        evaluate_competitor(bench, exp_path, n_jobs, verbose)
