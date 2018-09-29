#!/usr/bin/env python3


import pandas as pd
import numpy as np
import random
import utils


class KMeans:
    """

    """
    def __init__(self, k=None, means=None):
        """

        Parameters
        ----------
        k
        means
        """
        self.k = k
        self.means = means

    @property
    def k(self, k):
        """

        Parameters
        ----------
        k

        Returns
        -------

        """
        if not isinstance(k, int) or k < 1:
            raise ValueError('Invalid value for number of clusters: %s' % k)
        self.k = k

    @property
    def means(self, means):
        """

        Parameters
        ----------
        means

        Returns
        -------

        """

    def train(self, inputs):
        """

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        # choose k random points as the initial means
        self.means = random.sample(inputs, self.k)

        assignments = None

        while True:
            new_assignments = [self._classify(v) for v in inputs]
            if assignments == new_assignments:
                break
            assignments = new_assignments

        # update the means

    def _classify(self, input):
        """

        Parameters
        ----------
        input

        Returns
        -------

        """
        sq_distances = [utils.squared_distance(input, w) for w in self.means]
        _, min_idx = min((dist, i) for i, dist in enumerate(sq_distances))
        return min_idx

def main():
    """

    Returns
    -------

    """
    args = parse_cmd_args()

    X = pd.read_csv(filepath_or_buffer=args.X, header=None)
    y = pd.read_csv(filepath_or_buffer=args.y, header=None)

    final_model = train_model()