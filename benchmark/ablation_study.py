import os
import sys

sys.path.insert(0, "../")

from benchmark.variants import evaluate_pca, evaluate_ica, evaluate_rp, evaluate_dist_average, \
    evaluate_dist_average_mSTAMP, evaluate_dist_average_min, evaluate_profile_average, \
    evaluate_profile_average_threshold, evaluate_profile_average_max, evaluate_cp_selection_clustering

from benchmark.utils import evaluate_candidate

import numpy as np

np.random.seed(1379)


# computes ablation study on given data set
def evaluate_competitor(dataset_name, exp_path, n_jobs, verbose):
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    competitors = [
        ("PCA", evaluate_pca),
        ("ICA", evaluate_ica),
        ("RP", evaluate_rp),
        ("DistAvg", evaluate_dist_average),
        ("mSTAMP", evaluate_dist_average_mSTAMP),
        ("MinDistAvg", evaluate_dist_average_min),
        ("ProfileAvg", evaluate_profile_average),
        ("ThresholdProfileAvg", evaluate_profile_average_threshold),
        ("MaxProfileAvg", evaluate_profile_average_max),
        ("CPClustering", evaluate_cp_selection_clustering),
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
