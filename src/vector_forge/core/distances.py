import numpy as np
import numpy.typing as npt

def squared_l2_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    
    """
    Calculate the squared L2 distance between two points.

    Parameters
    ----------
    a : array-like
        First point in n-dimensional space.
    b : array-like
        Second point in n-dimensional space.

    Returns
    -------
    float
        The Euclidean distance between points a and b.
    """
     
    # Convert input arrays to numpy arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # validate vector shapes
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    # validate dimensions
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    
    # validate empty arrays
    if a.size == 0 or b.size == 0:
        raise ValueError("Input arrays must not be empty.")
    
    # rejects NaN values and infinite values
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Input arrays must not contain NaN values.")
    
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("Input arrays must not contain infinite values.")

    # calculate sqwared L2 distance
    difference = a - b
    distance = np.dot(difference, difference)
    
    return float(distance)