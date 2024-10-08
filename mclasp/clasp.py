import os

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from numba.typed.typedlist import List
from sklearn.exceptions import NotFittedError

from mclasp.nearest_neighbour import KSubsequenceNeighbours
from mclasp.nearest_neighbour import cross_val_labels
from mclasp.scoring import map_scores
from mclasp.utils import check_input_time_series, check_excl_radius, numba_cache_safe
from mclasp.validation import map_validation_tests


@njit(fastmath=True, cache=False)
def _profile(offsets, start, end, window_size, score):
    """
    Computes the classification score profile given nearest neighbour offsets.

    Parameters
    ----------
    offsets : np.ndarray
        An array of shape (n_timepoints, k_neighbors) containing the offsets of the k nearest
        neighbors for each timepoint.
    start : int
        The first index to consider (inclusive).
    end : int
        The last index to consider (exclusive).
    window_size : int
        The size of the window used to calculate nearest neighbours.
    score : callable
        A callable that computes the score of a segmentation given the true and predicted labels.
        The callable must accept two arguments: y_true and y_pred, which are arrays of binary labels
        indicating whether each timepoint belongs to the first or second segment of the segmentation.

    Returns
    -------
    np.ndarray
        An array of shape (end-start,) containing the classification score profile.
    """
    profile = np.full(shape=end - start, fill_value=-np.inf, dtype=np.float64)

    for split_idx in range(start, end):
        y_true, y_pred = cross_val_labels(offsets, split_idx, window_size)
        profile[split_idx - start] = score(y_true, y_pred)

    return profile


@njit(fastmath=True, cache=True, parallel=True)
def _parallel_profile(offsets, pranges, window_size, score):
    """
    Computes the classification score profile given nearest neighbour offsets in parallel
    with n_jobs threads.

    Parameters
    ----------
    offsets : np.ndarray
        An array of shape (n_timepoints, k_neighbors) containing the offsets of the k nearest
        neighbors for each timepoint.
    pranges : ndarray of shape (m, 2), where each row is (start, end)
        Ranges in which the profile scpres are calculated per thread. Infers the number of threads.
    window_size : int
        The size of the window used to calculate nearest neighbours.
    score : callable
        A callable that computes the score of a segmentation given the true and predicted labels.
        The callable must accept two arguments: y_true and y_pred, which are arrays of binary labels
        indicating whether each timepoint belongs to the first or second segment of the segmentation.

    Returns
    -------
    np.ndarray
        An array of shape (n_timepoints,) containing the classification score profile.
    """
    profile = np.full(shape=offsets.shape[0], fill_value=-np.inf, dtype=np.float64)

    for idx in prange(len(pranges)):
        start, end = pranges[idx]
        profile[start:end] = _profile(offsets, start, end, window_size, score)

    return profile


class ClaSP:
    """
    An implementation of the ClaSP algorithm for detecting change points in time series data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window used for computing distances and offsets, by default 10.
    k_neighbours : int, optional
        The number of nearest neighbors to consider when computing distances and offsets, by default 3.
    distance: str
        The name of the distance function to be computed for determining the k-NNs. Available options are
        "znormed_euclidean_distance" and "euclidean_distance".
    score : str or callable, optional
        The name of the classification score to use.
        Available options are "roc_auc", "f1", by default "roc_auc".
    aggregation : str, optional
        The name of the aggregation type used for multivariate time series.
        Available options are "dist" for distance averaging, "score" for profile averaging, by default "dist".
    excl_radius : int, optional
        The radius of the exclusion zone around the detected change point, by default 5*window_size.
    n_jobs : int, optional (default=1)
        Amount of threads used in the ClaSP computation.

    Methods
    -------
    fit(time_series)
        Create a ClaSP for the input time series data.
    predict()
        Return the ClaSP for the input time series data.
    fit_predict(time_series)
        Create and return a ClaSP for the input time series data.
    split()
        Split ClaSP into two segments.
    """

    def __init__(self, window_size=10, k_neighbours=3, distance="znormed_euclidean_distance", score="roc_auc",
                 aggregation="dist", excl_radius=5, n_jobs=-1):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.distance = distance
        self.score_name = score
        self.score = map_scores(score)
        self.aggregation = aggregation
        self.excl_radius = excl_radius
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.is_fitted = False

        check_excl_radius(k_neighbours, excl_radius)

    def _check_is_fitted(self):
        """
        Checks if the ClaSP object is fitted.

        Raises
        ------
        NotFittedError
            If the ClaSP object is not fitted.

        Returns
        -------
        None
        """
        if not self.is_fitted:
            raise NotFittedError("ClaSP object is not fitted yet. Please fit the object before using this method.")

    def fit(self, time_series, knns=None, validation="significance_test", threshold=1e-30):
        """
        Fits the ClaSP model to the input time series data.

        Parameters
        ----------
        time_series : numpy.ndarray
            The input time series data to fit the model on.

        knns : List of KSubsequenceNeighbours, optional
            Pre-computed KSubsequenceNeighbours objects (one for each channel) to use for fitting the model.
            If None (default), new KSubsequenceNeighbours objects will be created
            and fitted on the input time series data.
        validation : str, optional
            The validation method to use for determining the significance of the change point
            when early stopping is activated. The available methods are "significance_test" and
            "score_threshold". Default is "significance_test".
        threshold : float, optional
            The threshold value to use for the validation test. If the validation method is
            "significance_test", this value represents the p-value threshold for rejecting the
            null hypothesis. If the validation method is "score_threshold", this value represents
            the threshold score for accepting the change point. Default is 1e-15.

        Returns
        -------
        self : ClaSP
            The fitted ClaSP object.

        Raises
        ------
        ValueError
            If the input time series has less than 2*min_seg_size data points.
        """
        time_series = check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius
        self.lbound, self.ubound = 0, time_series.shape[0]

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series

        if knns is None:
            knns = []

            if self.aggregation.startswith("dist"):
                knns.append(
                    KSubsequenceNeighbours(
                        window_size=self.window_size,
                        k_neighbours=self.k_neighbours,
                        distance=self.distance,
                        aggregation=self.aggregation,
                        n_jobs=self.n_jobs
                    ).fit(time_series)
                )
            elif self.aggregation.startswith("score"):
                for dim in range(time_series.shape[1]):
                    knns.append(
                        KSubsequenceNeighbours(
                            window_size=self.window_size,
                            k_neighbours=self.k_neighbours,
                            distance=self.distance,
                            aggregation=self.aggregation,
                            n_jobs=self.n_jobs
                        ).fit(time_series[:,dim])
                    )
            else:
                raise ValueError(f"{self.aggregation} is not a valid aggregation method.")

        pranges = List()
        n_jobs = self.n_jobs
        n_offsets = knns[0].offsets.shape[0]

        while n_offsets // n_jobs < self.min_seg_size and n_jobs != 1:
            n_jobs -= 1

        bin_size = n_offsets // n_jobs

        for idx in range(n_jobs):
            start = max(idx * bin_size, self.min_seg_size)
            end = min((idx + 1) * bin_size, n_offsets - self.min_seg_size + self.window_size)
            if end > start: pranges.append((start, end))

        n_threads = get_num_threads()
        set_num_threads(n_jobs)

        if self.aggregation.startswith("dist"):
            self.profile = numba_cache_safe(_parallel_profile, knns[0].offsets, pranges, self.window_size, self.score)
        elif self.aggregation.startswith("score"):
            profiles = np.full(shape=(time_series.shape[1], n_offsets), fill_value=-np.inf, dtype=np.float64)

            for idx, knn in enumerate(knns):
                profiles[idx] = numba_cache_safe(_parallel_profile, knn.offsets, pranges, self.window_size, self.score)

            # profile averaging with significant profiles
            if self.aggregation == "score_threshold":
                mask = np.full(profiles.shape[0], fill_value=True, dtype=bool)
                self.is_fitted = True

                for idx, knn in enumerate(knns):
                    self.knns = [knn]
                    self.profile = profiles[idx]
                    mask[idx] = self.split(validation=validation, threshold=threshold) is not None

                knns = [knn for idx, knn in enumerate(knns) if mask[idx]]
                self.profile = profiles[mask].mean(axis=0)

            # profile averaging selecting ones with maximal scores
            elif self.aggregation == "score_max":
                maxima = np.array([np.max(p) for p in profiles])
                args = np.argsort(maxima)[::-1]

                knns = [knn for idx, knn in enumerate(knns) if idx in args[:time_series.shape[1] // 2]]
                self.profile = profiles[args][:time_series.shape[1] // 2].mean(axis=0)
            else:
                self.profile = profiles.mean(axis=0)
        else:
            raise ValueError(f"{self.aggregation} is not a valid aggregation method.")

        set_num_threads(n_threads)

        self.knns = knns
        self.is_fitted = True
        return self

    def transform(self):
        """
        Transform the input time series into a ClaSP profile.

        Returns
        -------
        profile : numpy.ndarray
            The ClaSP profile for the input time series.
        """
        self._check_is_fitted()
        return self.profile

    def fit_transform(self, time_series, knns=None, validation="significance_test", threshold=1e-30):
        """
        Fit the ClaSP algorithm to the given time series and return the
        corresponding profile.

        Parameters
        ----------
        time_series : np.ndarray, shape (n_timepoints,) or (n_timepoints, d_dimensions)
            The input time series to be segmented.
        knns : List of KSubsequenceNeighbours, optional
            Pre-computed KSubsequenceNeighbours objects (one for each channel) to use for fitting the model.
            If None (default), new KSubsequenceNeighbours objects will be created
            and fitted on the input time series data.
        validation : str, optional
            The validation method to use for determining the significance of the change point
            when early stopping is activated. The available methods are "significance_test" and
            "score_threshold". Default is "significance_test".
        threshold : float, optional
            The threshold value to use for the validation test. If the validation method is
            "significance_test", this value represents the p-value threshold for rejecting the
            null hypothesis. If the validation method is "score_threshold", this value represents
            the threshold score for accepting the change point. Default is 1e-15.

        Returns
        -------
        np.ndarray, shape (n_timepoints,)
            The ClaSP scores corresponding to each time point of the input time series.

        """
        return self.fit(time_series, knns, validation, threshold).transform()

    def split(self, sparse=True, validation="significance_test", threshold=1e-30):
        """
        Split the input time series into two segments using the change point location.

        Parameters
        ----------
        sparse : bool, optional
            If True, returns only the index of the change point. If False, returns the two segments
            separated by the change point. Default is True.
        validation : str, optional
            The validation method to use for determining the significance of the change point.
            The available methods are "significance_test" and "score_threshold". Default is
            "significance_test".
        threshold : float, optional
            The threshold value to use for the validation test. If the validation method is
            "significance_test", this value represents the p-value threshold for rejecting the
            null hypothesis. If the validation method is "score_threshold", this value represents
            the threshold score for accepting the change point. Default is 1e-15.

        Returns
        -------
        int or tuple
            If `sparse` is True, returns the index of the change point. If False, returns a tuple
            of the two time series segments separated by the change point.

        Raises
        ------
        ValueError
            If the `validation` parameter is not one of the available methods.
        """
        self._check_is_fitted()
        cp = np.argmax(self.profile)

        if validation is not None:
            validation_test = map_validation_tests(validation)
            if not validation_test(self, cp, threshold): return None

        if sparse is True:
            return cp

        return self.time_series[:cp], self.time_series[cp:]


class ClaSPEnsemble(ClaSP):
    """
    An ensemble of ClaSP.

    Parameters
    ----------
    n_estimators : int, optional
        The number of ClaSP models to use in the ensemble. Default is 10.
    window_size : int, optional
        The size of the window used for the k-subsequence neighbours. Default is 10.
    k_neighbours : int, optional
        The number of nearest neighbours to consider in the k-subsequence method. Default is 3.
    distance: str
        The name of the distance function to be computed for determining the k-NNs. Available options are
        "znormed_euclidean_distance" and "euclidean_distance".
    score : str or callable, optional
        The scoring method to use in the profile scoring. Must be a string ("f1" "roc_auc",).
        Default is "roc_auc".
    early_stopping : bool
        Determines if ensembling is stopped, once a validated change point is found or
        the ClaSP models do not improve anymore. Default is True.
    aggregation : str, optional
        The name of the aggregation type used for multivariate time series.
        Available options are "dist" for distance averaging, "score" for profile averaging, by default "dist".
    excl_radius : int, optional
        The radius of the exclusion zone in the profile scoring. Default is 5*window_size.
    n_jobs : int, optional (default=1)
        Amount of threads used in the ClaSP computation.
    random_state : int or RandomState, optional
        Seed for the random number generator. Default is 2357.

    Methods
    -------
    fit(time_series)
        Create a ClaSP ensemble for the input time series data.
    predict()
        Return the ClaSP ensemble for the input time series data.
    fit_predict(time_series)
        Create and return a ClaSP ensemble for the input time series data.
    """

    def __init__(self, n_estimators=10, window_size=10, k_neighbours=3, distance="znormed_euclidean_distance",
                 score="roc_auc", early_stopping=True, aggregation="dist", excl_radius=5, n_jobs=-1, random_state=2357):
        super().__init__(window_size, k_neighbours, distance, score, aggregation, excl_radius, n_jobs)
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.random_state = random_state

    def _calculate_temporal_constraints(self):
        """
        Calculates a set of random temporal constraints for each ClaSP in the ensemble.

        Returns
        -------
        tcs : ndarray of shape (n_estimators, 2)
            Array of start and end indices for each temporal constraint.
        """
        tcs = [(0, self.time_series.shape[0])]
        np.random.seed(self.random_state)

        while len(tcs) < self.n_estimators and self.time_series.shape[0] > 3 * self.min_seg_size:
            lbound, area = np.random.choice(self.time_series.shape[0], 2, replace=True)

            if self.time_series.shape[0] - lbound < area:
                area = self.time_series.shape[0] - lbound

            ubound = lbound + area
            if ubound - lbound < 2 * self.min_seg_size: continue
            tcs.append((lbound, ubound))

        return np.asarray(sorted(tcs, key=lambda tc: tc[1] - tc[0], reverse=True), dtype=np.int64)

    def fit(self, time_series, knns=None, validation="significance_test", threshold=1e-30):
        """
        Fits the ClaSP ensemble on the given time series, using temporal constraints to so that
        each ClaSP instance works on different (but possibly overlapping) parts of the time series.

        Parameters
        ----------
        time_series : np.ndarray
            The input time series of shape (n_samples,) or (n_samples, d_dimensions)
        knns : List of KSubsequenceNeighbours, optional
            Pre-computed KSubsequenceNeighbours objects (one for each channel) to use for fitting the model.
            If None (default), new KSubsequenceNeighbours objects will be created
            and fitted on the input time series data.
        validation : str, optional
            The validation method to use for determining the significance of the change point
            when early stopping is activated. The available methods are "significance_test" and
            "score_threshold". Default is "significance_test".
        threshold : float, optional
            The threshold value to use for the validation test. If the validation method is
            "significance_test", this value represents the p-value threshold for rejecting the
            null hypothesis. If the validation method is "score_threshold", this value represents
            the threshold score for accepting the change point. Default is 1e-15.

        Returns
        -------
        ClaSPEnsemble
            The fitted ClaSPEnsemble object.

        Raises
        ------
        ValueError
            If the input time series has less than 2 times the minimum segment size.
        """
        time_series = check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series
        tcs = self._calculate_temporal_constraints()

        if knns is None:
            knns = []

            if self.aggregation.startswith("dist"):
                knns.append(
                    KSubsequenceNeighbours(
                        window_size=self.window_size,
                        k_neighbours=self.k_neighbours,
                        distance=self.distance,
                        aggregation=self.aggregation,
                        n_jobs=self.n_jobs
                    ).fit(time_series, temporal_constraints=tcs)
                )
            elif self.aggregation.startswith("score"):
                for dim in range(time_series.shape[1]):
                    knns.append(
                        KSubsequenceNeighbours(
                            window_size=self.window_size,
                            k_neighbours=self.k_neighbours,
                            distance=self.distance,
                            aggregation=self.aggregation,
                            n_jobs=self.n_jobs
                        ).fit(time_series[:, dim], temporal_constraints=tcs)
                    )
            else:
                raise ValueError(f"{self.aggregation} is not a valid aggregation method.")

        best_score, best_tc, best_clasp = -np.inf, None, None

        for idx, (lbound, ubound) in enumerate(tcs):
            constrained_knns = [knn.constrain(lbound, ubound) for knn in knns]

            clasp = ClaSP(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                score=self.score_name,
                aggregation=self.aggregation,
                excl_radius=self.excl_radius,
                n_jobs=self.n_jobs
            ).fit(time_series[lbound:ubound], knns=constrained_knns, validation=validation, threshold=threshold)

            clasp.profile = (clasp.profile + (ubound - lbound) / time_series.shape[0]) / 2

            if clasp.profile.max() > best_score or best_clasp is None and idx == tcs.shape[0] - 1:
                best_score = clasp.profile.max()
                best_tc = (lbound, ubound)
                best_clasp = clasp
            else:
                if self.early_stopping is True: break

            if self.early_stopping is True and best_clasp.split(validation=validation, threshold=threshold) is not None:
                break

        self.profile = np.full(shape=time_series.shape[0] - self.window_size + 1, fill_value=-np.inf, dtype=np.float64)

        if best_clasp is not None:
            self.knns = best_clasp.knns
            self.lbound, self.ubound = best_tc
            self.profile[self.lbound:self.ubound - self.window_size + 1] = best_clasp.profile
        else:
            self.knns = knns
            self.lbound, self.ubound = 0, self.time_series.shape[0]

        self.is_fitted = True
        return self
