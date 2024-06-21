import daproli as dp
import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark.metrics import covering, f_measure
from src.utils import load_has_datasets


# evaluates results from segmentation algorithm
def evalute_segmentation_algorithm(dataset, n_timestamps, cps_true, cps_pred, profile=None):
    f1_score = np.round(f_measure({0: cps_true}, cps_pred, margin=int(n_timestamps * .01)), 3)
    covering_score = np.round(covering({0: cps_true}, cps_pred, n_timestamps), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score} Found CPs: {cps_pred}")

    if profile is not None:
        return dataset, cps_true.tolist(), cps_pred.tolist(), f1_score, covering_score, profile.tolist()

    return dataset, cps_true.tolist(), cps_pred.tolist(), f1_score, covering_score


# evaluates competitor on given data set
def evaluate_candidate(dataset_name, candidate_name, eval_func, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    if dataset_name == "HAS":
        df = load_has_datasets()
    elif dataset_name == "HAS_train":
        df = load_has_datasets(split="public")
    elif dataset_name == "HAS_test":
        df = load_has_datasets(split="private")
    else:
        df = load_datasets(dataset_name)

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