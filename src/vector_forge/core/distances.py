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


def pairwise_squared_l2(
    vectors : npt.ArrayLike,
    query: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """
    Calculate the squared L2 distance between a query point and a set of vectors.

    Parameters
    ----------
    vectors : array-like
        A 2D array where each row is a point in n-dimensional space.
    query : array-like
        A 1D array representing the query point in n-dimensional space.

    Returns
    -------
    np.ndarray
        A 1D array containing the squared L2 distances from the query point to each vector.
    """
    
    # Convert input arrays to numpy arrays
    vectors = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    
    # validate vector shapes
    if vectors.ndim != 2:
        raise ValueError("Input vectors must be a 2D array.")
    
    if query.ndim != 1:
        raise ValueError("Query point must be a 1D array.")
    
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("The number of dimensions of the query point must match the number of columns in the vectors.")
    
    # validate empty arrays
    if vectors.size == 0 or query.size == 0:
        raise ValueError("Input arrays must not be empty.")
    
    # rejects NaN values and infinite values
    if np.isnan(vectors).any() or np.isnan(query).any():
        raise ValueError("Input arrays must not contain NaN values.")
    
    if np.isinf(vectors).any() or np.isinf(query).any():
        raise ValueError("Input arrays must not contain infinite values.")

    # calculate squared L2 distances
    differences = vectors - query
    distances = np.sum(differences * differences, axis=1)
    
    return distances