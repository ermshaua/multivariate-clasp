import logging, sys
sys.path.insert(0, "../")

from mclasp.segmentation import MultivariateClaSPSegmentation


from benchmark.metrics import covering, f_measure
from src.utils import load_has_datasets

import numpy as np

np.random.seed(1379)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    df_data = load_has_datasets()

    idx = 0
    dataset, w, cps_true, labels, ts = df_data.iloc[idx, :]
    n_timestamps = ts.shape[0]

    clasp = MultivariateClaSPSegmentation()
    cps_pred = clasp.fit_predict(ts)

    f1_score = np.round(f_measure({0: cps_true}, cps_pred, margin=int(n_timestamps * .01)), 3)
    covering_score = np.round(covering({0: cps_true}, cps_pred, n_timestamps), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score} Found CPs: {cps_pred}")
