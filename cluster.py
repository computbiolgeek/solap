#!/usr/bin/env python3

from argparse import ArgumentParser
import random
import utils
import matplotlib.pyplot as plt
import logging
from sklearn import cluster
import math


logging.basicConfig(
    level=logging.INFO
)


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
    def k(self):
        return self._k

    @k.setter
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
        self._k = k

    @property
    def means(self):
        return self._means

    @means.setter
    def means(self, means):
        """

        Parameters
        ----------
        means

        Returns
        -------

        """
        self._means = means

    def train(self, inputs, initial_means=None):
        """

        Parameters
        ----------
        inputs : list
            A list of lists, each consists of a pair of x, y coordinates.
        initial_means : list
            A list of lists, each consists of a pair of x, y coordinates,
            as initial means.

        Returns
        -------

        """
        if initial_means is not None:
            logging.info('Using initial means.')
            self.means = initial_means
        else:
            # choose k random points as the initial means
            logging.info('Using random starting means.')
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
            clusters = [[] for _ in range(self.k)]
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


def standardize(inputs):
    """Standardize features according to (x - u) / s where u is mean and s
    is standard deviation.

    Parameters
    ----------
    inputs : list
        A list of lists of the dimension m * n where m is the number of
        samples and n is the number of features each sample is described.

    Returns
    -------
    list
        Standardized features.

    """
    mean_vector = utils.vector_mean(inputs)
    squares = []
    for p in inputs:
        squares.append(
            [(val - mean_vector[i]) ** 2 for i, val in enumerate(p)]
        )
    stds = [math.sqrt(x) for x in utils.scalar_multiply(1/len(
        squares), utils.vector_sum(squares))]
    standardized_inputs = []
    for p in inputs:
        standardized_inputs.append(
            [(val - mean_vector[i]) / stds[i] for i, val in enumerate(p)]
        )
    return standardized_inputs


def dbscan(X, eps=0.3, min_samples=10):
    """

    Parameters
    ----------
    X
    eps
    min_samples

    Returns
    -------

    """
    X_standardized = standardize(X)
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X_standardized)
    return db.labels_


def parse_cmd_args():
    """

    Returns
    -------

    """
    parser = ArgumentParser(description='K-Means clustering.')
    parser.add_argument('--input', '-i', dest='input', required=True,
                        type=str, help='Input points.')
    parser.add_argument('--k', '-k', dest='k', required=False, type=int,
                        help='The number of clusters.')
    parser.add_argument('--means', '-m', dest='means', required=False,
                        type=str, help='Initial means.')
    parser.add_argument('--algorithm', '-a', dest='algorithm', type=str,
                        required=True, help='Clustering algorithm. Choose one '
                        'from KMEANS, DBSCAN, HIER')
    parser.add_argument('--prefix', '-p', dest='prefix', type=str,
                        required=True, help='Prefix for output file names.')
    return parser.parse_args()


def main():
    """

    Returns
    -------

    """
    args = parse_cmd_args()
    with open(args.input, 'rt') as ipf:
        points = [[float(f) for f in line.strip().split(',')] for line in ipf]

    if args.algorithm.upper() == 'KMEANS':
        if args.means is not None:
            with open(args.means, 'rt') as ipf:
                initial_means = [
                    [float(f) for f in line.strip().split(',')] for line in ipf
                ]
        else:
            initial_means = None
        k_means = KMeans(k=args.k)
        assignments = k_means.train(points, initial_means)
    if args.algorithm.upper() == 'DBSCAN':
        assignments = dbscan(points, eps=0.3, min_samples=3)
        n_clusters = len(set(assignments)) - (1 if -1 in assignments else 0)
        logging.info('Number of clusters: %d' % n_clusters)

    with open(args.prefix + '_assignments.txt', 'wt') as opf:
        opf.writelines('%s\n' % a for a in assignments)

    # generate a plot
    xs = [xy[0] for xy in points]
    ys = [xy[1] for xy in points]
    c_list= ['violet', 'blue', 'skyblue', 'steelblue', 'cyan', 'green', 'teal',
             'orange', 'red', 'grey', 'black']
    colors = [c_list[l] for l in assignments]
    fig = plt.figure()
    plt.scatter(xs, ys, c=colors)
    fig.savefig(args.prefix + '_plot_clusters.pdf')


if __name__ == '__main__':
    main()
