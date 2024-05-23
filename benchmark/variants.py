from sklearn.cluster import AgglomerativeClustering

from mclasp.segmentation import MultivariateClaSPSegmentation
from benchmark.utils import evalute_segmentation_algorithm
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

def evaluate_profile_average(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = MultivariateClaSPSegmentation(aggregation="score", n_jobs=4)
    cps_pred = clasp.fit_predict(ts)
    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)

def evaluate_profile_average_threshold(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = MultivariateClaSPSegmentation(aggregation="score_threshold", n_jobs=4)
    cps_pred = clasp.fit_predict(ts)
    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)

def evaluate_profile_average_max(dataset, w, cps_true, labels, ts, **seg_kwargs):
    clasp = MultivariateClaSPSegmentation(aggregation="score_max", n_jobs=4)
    cps_pred = clasp.fit_predict(ts)
    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)

def evaluate_cp_selection_clustering(dataset, w, cps_true, labels, ts, **seg_kwargs):
    profiles, found_cps, window_sizes = [], [], []

    for time_series in ts.T:
        clasp = BinaryClaSPSegmentation(n_jobs=4)
        found_cps.extend(clasp.fit_predict(time_series))
        profiles.append(clasp.profile)
        window_sizes.append(clasp.window_size)

    found_cps = np.array(found_cps)

    if len(found_cps) < ts.shape[1] // 2:
        cps_pred = np.zeros(0, int)
    else:
        clu = AgglomerativeClustering(n_clusters=None, linkage="average", distance_threshold=5 * np.mean(window_sizes))
        clusters = clu.fit_predict(found_cps.reshape(-1, 1))

        merged_cps = []

        for label in np.unique(clusters):
            candidates = found_cps[clusters == label]

            if len(candidates) < ts.shape[1] // 2: continue
            merged_cps.append(int(np.mean(candidates)))

        cps_pred = np.sort(merged_cps)

    return evalute_segmentation_algorithm(dataset, ts.shape[0], cps_true, cps_pred)