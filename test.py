#!/usr/bin/env python3

import logging
from sklearn.metrics import mean_squared_error
import numpy as np

# for writing diagnostic information to log files
logging.basicConfig(level=logging.INFO)


def test_model(X, y, model=None):
    """

    Parameters
    ----------
    X
    y
    model

    Returns
    -------
    float
        Root-mean-square error.

    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print('RMSE on independet test set: %.2f' % np.sqrt(mse))
    return predictions
