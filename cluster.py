#!/usr/bin/env python3

import numpy as np
from argparse import ArgumentParser
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
            # assign each point to a cluster
            new_assignments = [self._classify(v) for v in inputs]
            # if no assignments have changed, we're done
            if assignments == new_assignments:
                break
            # otherwise, keep new assignments and update the means
            assignments = new_assignments
            clusters = [[] for _ in self.k]
            for i, j in enumerate(assignments):
                clusters[j].append(inputs[i])
            for i, c in enumerate(clusters):
                if c:
                    self.means[i] = utils.vector_mean(c)
        return assignments

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


def parse_cmd_args():
    """

    Returns
    -------

    """
    parser = ArgumentParser(description='K-Means clustering.')
    parser.add_argument('--input', '-i', dest='input', required=True,
                        type=str, help='Input points.')
    parser.add_argument('--output', '-o', dest='output', required=True,
                        type=str, help='Cluster assignments.')
    parser.add_argument('--k', '-k', dest='k', required=True, type=int,
                        help='The number of clusters.')
    return parser.parse_args()


def main():
    """

    Returns
    -------

    """
    args = parse_cmd_args()
    points = np.loadtxt(args.input, delimiter=',')
    k_means = KMeans(k=args.k)
    assignments = k_means.train(points)
    with open(args.output, 'wt') as opf:
        opf.writelines('%s\n' % a for a in assignments)