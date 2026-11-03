import numpy as np
import pytest

from src.vector_forge.core.distances import pairwise_squared_l2, squared_l2_distance

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
    
    
    
def test_pairwise_squared_l2():
    vectors = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 3, 4],
    ]

    query = [1, 1, 1]

    result = pairwise_squared_l2(vectors, query)

    expected = np.array([5, 50, 149, 14])

    np.testing.assert_allclose(result, expected)



def test_pairwise_output_shape():
    vectors = np.random.rand(10, 5)
    query = np.random.rand(5)

    result = pairwise_squared_l2(vectors, query)

    assert result.shape == (10,)


def test_pairwise_single_vector():
    vectors = [[1, 2, 3]]
    query = [1, 1, 1]

    result = pairwise_squared_l2(vectors, query)

    np.testing.assert_allclose(result, [5.0])


def test_pairwise_dimension_mismatch():
    vectors = [[1, 2, 3]]
    query = [1, 2]

    with pytest.raises(ValueError):
        pairwise_squared_l2(vectors, query)


def test_pairwise_empty_dataset():
    vectors = np.empty((0, 3))
    query = [1, 2, 3]

    with pytest.raises(ValueError):
        pairwise_squared_l2(vectors, query)


def test_pairwise_nan():
    vectors = [[1, np.nan, 3]]
    query = [1, 2, 3]

    with pytest.raises(ValueError):
        pairwise_squared_l2(vectors, query)
    

def test_pairwise_infinity():
    vectors = [[1, np.inf, 3]]
    query = [1, 2, 3]

    with pytest.raises(ValueError):
        pairwise_squared_l2(vectors, query)


def test_pairwise_matches_individual_distance():
    vectors = np.random.default_rng(42).random((20, 5))
    query = np.random.default_rng(43).random(5)

    result = pairwise_squared_l2(vectors, query)

    expected = np.array([
        squared_l2_distance(vector, query)
        for vector in vectors
    ])

    np.testing.assert_allclose(result, expected)