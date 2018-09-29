#!/usr/bin/env python3

import logging
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sknn.mlp import Layer, Regressor, Classifier


# for writing diagnostic information to log files
logging.basicConfig(level=logging.INFO)


def train_model(X, y, model_type, cv_fold=5):
    """

    Parameters
    ----------
    X : NumPy ndarray
        The design matrix.
    y : NumPy array
        Target labels.
    model_type : str
        'classifier' or 'regressor'
    cv_fold : int
        Fold of cross-validation.
    Returns
    -------
    Estimator
        Best estimator obtained from grid searching.

    """
    # a grid of hyperparameters from which to search for an optimal combination
    param_grid = {
        'alpha': [0.1, 0.05, 0.01, 0.005, 0.001,0.0001, 0.00001],
        'hidden_layer_sizes': [(8,), (16,), (32,), (64,)],
        'momentum': np.arange(0.1, 1.0, 0.1)
    }

    # get model type
    if model_type == 'classifier':
        model = MLPClassifier(verbose=True)
    else:
        model = MLPRegressor(verbose=True)

    # do a grid search for optimal hyperparameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv_fold,
        refit=True
    )
    logging.info('Fitting model with grid search')
    grid_search.fit(X, y)

    # print results from grid search
    logging.info('best hyperparameter combination %s' % grid_search.best_params_)
    gs_results = grid_search.cv_results_
    for params, mean_score in zip(
        gs_results['params'], gs_results['mean_test_score']
    ):
        print(params, '%.2f' % np.sqrt(-mean_score))

    # return the final model
    return grid_search.best_estimator_


def train_dropout_nn(X, y, model_type='classifier', cv_fold=5):
    """

    Parameters
    ----------
    X
    y
    model_type
    cv_fold

    Returns
    -------

    """
    # a grid of hyperparameters from which to search for an optimal combination
    param_grid = {
        'weight_decay': [0.05, 0.01, 0.005, 0.001],
        'dropout_rate': [0.25, 0.50],
        'learning_momentum': np.arange(0.1, 1.0, 0.3),
        'learning_rate': [0.05, 0.01, 0.005, 0.001],
        'hidden0__units': [8, 16, 32, 64],
        'hidden0__dropout': [0.25, 0.50]
    }

    # create appropriate model type
    if model_type == 'classifier':
        model = Classifier(
            layers=[Layer('Sigmoid'), Layer('Softmax')],
            regularize='L2',
            verbose=True
        )
    else:
        model = Regressor(
            layers=[Layer('Sigmoid'), Layer('Linear')],
            regularize='L2',
            verbose=True
        )

    # do a grid search for optimal hyperparameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv_fold,
        refit=True
    )
    logging.info('Fitting neural networks regularized with dropout ...')
    grid_search.fit(X, y)

    # print results from grid search
    logging.info('best hyperparameter combination %s' % grid_search.best_params_)
    gs_results = grid_search.cv_results_
    for params, mean_score in zip(
        gs_results['params'], gs_results['mean_test_score']
    ):
        print(params, '%.2f' % np.sqrt(-mean_score))

    # return the final model
    return grid_search.best_estimator_
