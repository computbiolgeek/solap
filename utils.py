#!/usr/bin/env python3

import math

def vector_add(v, w):
    """

    Parameters
    ----------
    v
    w

    Returns
    -------

    """
    if len(v) != len(w):
        raise ValueError('Incompatible vector lengths: %s vs %s'
                         % (len(v), len(w)))
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_subtract(v, w):
    """

    Parameters
    ----------
    v
    w

    Returns
    -------

    """
    if len(v) != len(w):
        raise ValueError('Incompatible vector lengths: %s vs %s'
                         % (len(v), len(w)))
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    """Compute a vector whose ith element is the sum of the ith elements of
    the input vectors.

    Parameters
    ----------
    vectors

    Returns
    -------
    list
        A list whose ith element is the sum of the ith elements of the input
        vectors.

    """
    result = vectors[0]
    for v in vectors[1:]:
        result = [a + b for a, b in zip(result, v)]
    return result


def scalar_multiply(c, v):
    """

    Parameters
    ----------
    c : float
        A number.
    v : list
        A list.

    Returns
    -------
    list
        A list whose ith element is the ith element of the input vector
        multiplied by c.

    """
    return [c * v_i for v_i in v]


def vector_mean(vectors):
    """Compute the vector whose ith element is the mean of the ith elements
    of the input vectors.

    Parameters
    ----------
    vectors

    Returns
    -------
    list
        A list whose ith element is the mean of the ith elements of the
        input vectors.

    """
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v, w):
    """

    Parameters
    ----------
    v
    w

    Returns
    -------

    """
    if len(v) != len(w):
        raise ValueError('Incompatible vector lengths: %s vs %s'
                         % (len(v), len(w)))
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    """Computes the sum of squares of each element in the input vector.

    Parameters
    ----------
    v : list
        A list.

    Returns
    -------
    float
        The sum of squares of each element in the input vector.

    """
    return dot(v, v)


def squared_distance(v, w):
    """

    Parameters
    ----------
    v
    w

    Returns
    -------

    """
    return sum_of_squares(vector_subtract(v, w))


def distance(v, w):
    """

    Parameters
    ----------
    v
    w

    Returns
    -------

    """
    return math.sqrt(squared_distance(v, w))