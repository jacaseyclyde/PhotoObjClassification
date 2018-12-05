# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC


CLASS_DICT = {}
SUBCLASS_DICT = {}


def _class_dict_init(classes):
    classes = np.unique(classes.astype(str))
    values = range(len(classes))

    global CLASS_DICT
    CLASS_DICT = dict(zip(classes, values))


def _subclass_dict_init(subclasses):
    subclasses = np.unique(subclasses.astype(str))
    values = range(len(subclasses))

    global SUBCLASS_DICT
    SUBCLASS_DICT = dict(zip(subclasses, values))


def load_data():
    # There is the possibility of this containing duplicate objects,
    # but because each row represents a seperate observation, it's ok
    # and we don't need to filter for them
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'data', 'sdss.csv'))

    # ID labels don't represent anything useful
    data = data.drop(labels=['objID', 'targetObjID', 'specObjID', 'obj',
                             'htmID', 'survey', 'programname', 'sourceType'],
                     axis=1)

    return data


def plot(X, y, n_dim=2):
    """Plots data projections with classes.

    Plots the projection of `n_dim` classes into each plane.

    """
    # plot the classes/colors
    fig1, ax1 = plt.subplots(n_dim, n_dim, sharex=True, sharey=True)
    fig1.suptitle('Corner Plot: First {0:.0f} dimensions'.format(n_dim))
    fig1.set_size_inches(4 * (n_dim - 1), 4 * (n_dim - 1))

    ax1[0, 0].set_xticklabels([])
    ax1[0, 0].set_yticklabels([])

    for i in range(n_dim - 1):
        for j in range(n_dim - 1):
            if j > i:
                ax1[i, j].axis('off')

            else:
                ax1[i, j].scatter(data[j], data[i + 1])

            if j == 0:
                ax1[i, j].set_ylabel("$p_{0:.0f}$".format(i + 1))

            if i == n_dim - 2:
                ax1[i, j].set_xlabel("$p_{0:.0f}$".format(j))

    fig1.subplots_adjust(hspace=0, wspace=0)

    recs = []
    for i in range(0, len(ckeys)):
        recs.append(mpatches.Circle((0, 0), radius=50, fc=cdict[ckeys[i]]))

    ax1[0, n_dim - 1].legend(recs, ckeys, loc="upper right", ncol=2)


def grid_search_optimizer(data, clf, params, components=.95, discr=None):
    X_train, X_test, y_train, y_test = data

    if components < 1:
        pca = PCA(n_components=components)
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    if discr is not None:
        discr.fit(X_train, y_train)

        X_train = discr.transform(X_train)
        X_test = discr.transform(X_test)

    clf_cv = GridSearchCV(clf, params, cv=10, n_jobs=-1)
    clf_cv.fit(X_train, y_train)

    err = 1 - clf_cv.score(X_test, y_test)

    return err, clf_cv.best_params_


def main():
    data = load_data()

    # Our two types of labels
    classes = data.pop('class')
    subclasses = data.pop('subClass')

    # initialize dicts relating integer class values to their human
    # friendly string values
    _class_dict_init(classes)
    _subclass_dict_init(subclasses)

    X = data.values
    y = np.vectorize(CLASS_DICT.get)(classes)

    # split data fro training/testing
    data_split = train_test_split(X, y, test_size=.25)

    results = pd.DataFrame(columns=['clf', 'err', 'pca',
                                    'lda', 'params'])
    clfs = [(KNeighborsClassifier(),
             "knn",
             {'n_neighbors': np.arange(25) + 1,
              'weights': ['uniform', 'distance']}),

            (LDA(),
             "lda",
             {}),

            (QDA(),
             "qda",
             {}),

            (SVC(),
             "svm",
             {'C': 2. ** np.arange(-6, 5),
              'gamma': 2. ** np.arange(-6, 5)})
            ]

    pca_vals = np.arange(.5, 1.05, .05)
    discriminants = [None, LDA()]

    n_tests = len(pca_vals) * len(discriminants) * len(clfs)
    with tqdm(total=n_tests) as pbar:
        for clf, label, params in clfs:
            for discriminant in discriminants:
                lda = isinstance(discriminant, LDA)

#                # skip lda with lda reduction, since thats redundant
#                if isinstance(clf, LDA) and lda:
#                    continue

                for components in pca_vals:  # variance retained
                    err, par = grid_search_optimizer(data_split, clf, params,
                                                     components=components,
                                                     discr=discriminant)

                    results = results.append(ignore_index=True,
                                             other={'clf': label,
                                                    'err': err,
                                                    'pca': components,
                                                    'lda': lda,
                                                    'params': par})
                    pbar.update()

    results.to_csv(os.path.join(os.path.dirname(__file__),
                                'out', 'results.csv'))

    return results


if __name__ == '__main__':
    figure = {'figsize': (12, 12)}
    mpl.rc('figure', **figure)

    font = {'size': 24}
    mpl.rc('font', **font)

    directory = os.path.join(os.path.dirname(__file__), 'out')
    if not os.path.exists(directory):
        os.makedirs(directory)
    results = main()
