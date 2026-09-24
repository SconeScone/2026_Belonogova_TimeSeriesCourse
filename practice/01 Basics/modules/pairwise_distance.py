import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series 

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    

    @property
    def distance_metric(self) -> str:
        """Return the distance metric

        Returns
        -------
            string with metric which is used to calculate distances between set of time series
        """

        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix
        
        Returns
        -------
        function reference
        """

        if self.metric == "euclidean":
            if self.is_normalize:
                return norm_ED_distance
            else:
                return ED_distance
        elif self.metric == "dtw":
            return DTW_distance
        else:
            raise RuntimeError("Unknown metric.")
                

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """ Calculate distance matrix
        
        Parameters
        ----------
        input_data: time series set
        
        Returns
        -------
        matrix_values: distance matrix
        """

        k = input_data.shape[0]
        matrix_values = np.zeros((k, k))

        if self.is_normalize and self.metric != "euclidean":
            input_data = np.array([z_normalize(ts) for ts in input_data])

        for i in range(k):
            for j in range(k):
                if i < j:
                    matrix_values[i, j] = self._choose_distance()(input_data[i], input_data[j])
                else:
                    matrix_values[i, j] = matrix_values[j, i]

        return matrix_values
