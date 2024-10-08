import sys
sys.path.insert(0, "../")

from mclasp.segmentation import MultivariateClaSPSegmentation


from benchmark.metrics import covering, f_measure
from src.utils import load_has_datasets

import numpy as np

np.random.seed(1379)


# computes an example with MClaSP
if __name__ == '__main__':
    df_data = load_has_datasets(split="public")

    idx = 0
    dataset, w, cps_true, labels, ts = df_data.iloc[idx, :]
    n_timestamps = ts.shape[0]

    clasp = MultivariateClaSPSegmentation()
    cps_pred = clasp.fit_predict(ts)

    f1_score = np.round(f_measure({0: cps_true}, cps_pred, margin=int(n_timestamps * .01)), 3)
    covering_score = np.round(covering({0: cps_true}, cps_pred, n_timestamps), 3)

    print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score} Found CPs: {cps_pred}")
    clasp.plot(gt_cps=cps_true, heading="Simple Test", ts_name=dataset, file_path="../tmp/simple_test.pdf")
