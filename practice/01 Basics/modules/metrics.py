import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    dif = ts1 - ts2
    ed_dist = np.sqrt(np.dot(dif, dif))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """
    
    n = len(ts1)

    mean1 = ts1.mean()
    std1 = ts1.std()
    
    mean2 = ts2.mean()
    std2 = ts2.std()
    
    norm_ed_dist = np.sqrt(np.abs(2 * n * (1 - (np.dot(ts1, ts2) - n * mean1 * mean2) / (n * std1 * std2))))

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """
    m = len(ts1)
    
    d = np.zeros((m, m))
    D = np.full((m + 1, m + 1), np.inf)
    D[0, 0] = 0
    
    for i in range(1, m + 1):
        for j in range(1, m + 1):
          d[i - 1, j - 1] = (ts1[i - 1] - ts2[j - 1])**2
          D[i, j] = d[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    dtw_dist = D[m, m]

    return dtw_dist
