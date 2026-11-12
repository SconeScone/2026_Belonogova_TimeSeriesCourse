import numpy as np

from modules.utils import *
import stumpy


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    mp = matrix_profile["mp"].copy()
    mpi =  matrix_profile["mpi"]
    excl_zone = matrix_profile["excl_zone"]

    for i in range(top_k):
        idx = np.argmin(mp)
        dist = mp[idx]
        nn_idx = mpi[idx]

        if idx <= nn_idx:
            left = idx
            right = nn_idx
        else:
            left = nn_idx
            right = idx

        motifs_idx.append((left, right))
        motifs_dist.append(dist)

        apply_exclusion_zone(mp, idx, excl_zone, np.inf)
        apply_exclusion_zone(mp, nn_idx, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
