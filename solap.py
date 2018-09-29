#!/usr/bin/env python3

import json
import pickle
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from test import test_model
from train import train_model


def parse_cmd_args():
    """
    Parses arguments given on the command line by the user.

    Returns
    -------
    ArgumentParser
        An object of type ArgumentParser.

    """
    parser = ArgumentParser(description='parses command-line arguments for '
                            'SOLAP')
    parser.add_argument('--train', '-t', dest='train', type=str, required=True,
                        help='Predictive features organized in a m * n design '
                             'matrix where m is the number of sample points '
                             'and n is the number of features')
    parser.add_argument('--eval', '-e', dest='eval', type=str, required=True,
                        help='The corresponding labels / target values for '
                             'the X matrix')
    parser.add_argument('--config', '-c', dest='config', type=str,
                        required=True, help='A configuration file in JSON '
                        'format.')
    parser.add_argument('--output', '-o', dest='output', type=str,
                        required=True, help='Output file where to store '
                        'predictions.')
    parser.add_argument('--model', '-m', dest='model', type=str,
                        required=True, help='Serialized final model.')
    args = parser.parse_args()
    # do command-line argument checking here if necessary
    return args


def main():
    """

    Returns
    -------

    """
    args = parse_cmd_args()

    df_train = pd.read_csv(filepath_or_buffer=args.train, header=None)
    df_test = pd.read_csv(filepath_or_buffer=args.eval, header=None)

    with open(args.config, 'rt') as ipf:
        indices = json.load(ipf)
    id_indices = eval(indices['id_indices'])
    x_indices = eval(indices['x_indices'])
    y_indices = eval(indices['y_indices'])

    X_train = df_train.iloc[:, x_indices].values
    y_train = df_train.iloc[:, y_indices].values
    X_test = df_test.iloc[:, x_indices].values
    y_test = df_test.iloc[:, y_indices].values
    ids_test = df_test.iloc[:, id_indices]

    final_model = train_model(
        X_train,
        y_train,
        model_type='regressor',
        cv_fold=5
    )

    # final_model = train_dropout_nn(
    #     X_train,
    #     y_train,
    #     model_type='regressor',
    #     cv_fold=5
    # )

    predictions = test_model(X_test, y_test, model=final_model)
    output = ids_test.assign(target=y_test, pred=predictions)

    with open(args.output, 'wt') as opf:
        output.to_csv(opf, float_format='%.2f', index=False, header=False)

    # refit the final model to the whole data set
    X_whole = np.concatenate((X_train, X_test))
    y_whole = np.concatenate((y_train, y_test))
    final_model.fit(X_whole, y_whole)

    # write final model to disk
    with open(args.model, 'wb') as opf:
        pickle.dump(final_model, opf)


if __name__ == '__main__':
    main()