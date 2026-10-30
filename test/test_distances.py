import numpy as np
import pytest

from src.vector_forge.core.distances import squared_l2_distance

def test_squared_l2_distance():
    result = squared_l2_distance(
    [1, 2, 3], 
    [4, 5, 6]
    )
    
    assert result == 27.0  # (4-1)^2 + (5-2)^2 + (6-3)^2 = 27  
    
    
def test_identical_vectors():
    result = squared_l2_distance(
    [1, 2, 3], 
    [1, 2, 3]
    )
    
    assert result == 0.0  # identical vectors should have distance of 0
    
def test_numpy_arrays():
    a = np.array([1, 2])
    b = np.array([4, 5])
    
    result = squared_l2_distance(a, b)
    
    assert result == 18.0  # (4-1)^2 + (5-2)^2 = 18
    
def test_different_dimensions():
    with pytest.raises(ValueError):
        squared_l2_distance([1, 2], [1, 2, 3])


def test_empty_vectors():
    with pytest.raises(ValueError):
        squared_l2_distance([], [])


def test_multidimensional_input():
    with pytest.raises(ValueError):
        squared_l2_distance(
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]]
        )


def test_nan_input():
    with pytest.raises(ValueError):
        squared_l2_distance([1, np.nan], [2, 3])


def test_infinity_input():
    with pytest.raises(ValueError):
        squared_l2_distance([1, np.inf], [2, 3])


def test_symmetry():
    a = [1, 2, 3]
    b = [4, 5, 6]

    assert squared_l2_distance(a, b) == squared_l2_distance(b, a)