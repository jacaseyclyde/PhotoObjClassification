# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


CLASS_DICT = {}
SUBCLASS_DICT = {}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
                                    'data', 'sdss.csv'),
                       nrows=500)

    # Get rid of features that don't represent physical data or
    # something directly derived from it (i.e., not a fit).
    # Commented features are currently considered as useful for
    # classification
    data = data.drop(labels=[
                             'specObjID',
#                             'mjd',
#                             'plate',
#                             'tile',
#                             'fiberID',
                             'z',
                             'zErr',
                             'zWarning',
#                             'ra',
#                             'dec',
                             'cx',
                             'cy',
                             'cz',
                             'htmID',
                             'sciencePrimary',
                             'legacyPrimary',
                             'seguePrimary',
                             'segue1Primary',
                             'segue2Primary',
                             'bossPrimary',
                             'sdssPrimary',
                             'survey',
                             'programname',
                             'legacy_target1',
                             'legacy_target2',
                             'special_target1',
                             'special_target2',
                             'segue1_target1',
                             'segue1_target2',
                             'segue2_target1',
                             'segue2_target2',
                             'boss_target1',
                             'ancillary_target1',
                             'ancillary_target2',
#                             'plateID',
                             'sourceType',
                             'targetObjID',
                             'objID',
#                             'skyVersion',
#                             'run',
#                             'rerun',
#                             'camcol',
#                             'field',
                             'obj',
#                             'mode',
#                             'nChild',
#                             'flags',
#                             'psfMag_u',
#                             'psfMag_g',
#                             'psfMag_r',
#                             'psfMag_i',
#                             'psfMag_z',
#                             'psfMagErr_u',
#                             'psfMagErr_g',
#                             'psfMagErr_r',
#                             'psfMagErr_i',
#                             'psfMagErr_z',
#                             'fiberMag_u',
#                             'fiberMag_g',
#                             'fiberMag_r',
#                             'fiberMag_i',
#                             'fiberMag_z',
#                             'fiberMagErr_u',
#                             'fiberMagErr_g',
#                             'fiberMagErr_r',
#                             'fiberMagErr_i',
#                             'fiberMagErr_z',
#                             'petroMag_u',
#                             'petroMag_g',
#                             'petroMag_r',
#                             'petroMag_i',
#                             'petroMag_z',
#                             'petroMagErr_u',
#                             'petroMagErr_g',
#                             'petroMagErr_r',
#                             'petroMagErr_i',
#                             'petroMagErr_z',
                             'modelMag_u',
                             'modelMag_g',
                             'modelMag_r',
                             'modelMag_i',
                             'modelMag_z',
                             'modelMagErr_u',
                             'modelMagErr_g',
                             'modelMagErr_r',
                             'modelMagErr_i',
                             'modelMagErr_z',
                             'cModelMag_u',
                             'cModelMag_g',
                             'cModelMag_r',
                             'cModelMag_i',
                             'cModelMag_z',
                             'cModelMagErr_u',
                             'cModelMagErr_g',
                             'cModelMagErr_r',
                             'cModelMagErr_i',
                             'cModelMagErr_z',
                             'mRrCc_r',
                             'mRrCcErr_r',
#                             'score',
#                             'resolveStatus',
#                             'calibStatus_u',
#                             'calibStatus_g',
#                             'calibStatus_r',
#                             'calibStatus_i',
#                             'calibStatus_z',
#                             'photoRa',
#                             'photoDec',
#                             'extinction_u',
#                             'extinction_g',
#                             'extinction_r',
#                             'extinction_i',
#                             'extinction_z',
#                             'fieldID',
                             'dered_u',
                             'dered_g',
                             'dered_r',
                             'dered_i',
                             'dered_z'
                             ],
                     axis=1)

    return data


def grid_search_optimizer(data, clf, params, components=None, lda=False):
    X_train, X_test, y_train, y_test = data

    if components != 'None':
        components = components.astype(float) / 100.
        if components < 1:
            pca = PCA(n_components=components)
            pca.fit(X_train)

            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        elif components == 1.:
            pca = PCA()
            pca.fit(X_train)

            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

    if lda:
        discr = LDA().fit(X_train, y_train)

        X_train = discr.transform(X_train)
        X_test = discr.transform(X_test)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf_cv = GridSearchCV(clf, params, cv=5, n_jobs=-1)
    clf_cv.fit(X_train, y_train)

    err = 1 - clf_cv.score(X_test, y_test)
    return err, clf_cv.best_params_


def analysis(data, tests):
    # string None b/c pandas indexing. it feels weird tbh
    pca_vals = np.concatenate((['None'],
                               (100 * np.arange(.95, 1.0, .01)).astype(int)))
    lda = [True, False]

#    results = pd.DataFrame(columns=['clf', 'err', 'pca',
#                                    'lda', 'params'])
    ind = pd.MultiIndex.from_product([pca_vals, lda, ['err', 'params']],
                                     names=['pca', 'lda', 'results'])
    results = pd.DataFrame(index=ind, columns=tests.columns)

    n_tests = len(pca_vals) * len(lda) * len(tests.columns)
    with tqdm(total=n_tests, file=sys.stdout) as pbar:
        for test in tests:
            clf = tests[test]['obj']
            params = tests[test]['grid']
            for lda_bool in lda:
                for pca in pca_vals:  # variance retained
                    pbar.set_description("clf: {0}, "
                                         "lda: {1}, "
                                         "pca: {2}".format(test,
                                                           lda_bool, pca))

                    err, par = grid_search_optimizer(data, clf, params,
                                                     components=pca,
                                                     lda=lda_bool)

                    results[test][pca, lda_bool, 'err'] = round(err, 4)
                    results[test][pca, lda_bool, 'params'] = par

                    pbar.update()

    results.to_csv(os.path.join(os.path.dirname(__file__),
                                'out', 'results.csv'))
    return results


def plot_results(results):
    pass


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

    # split data for training/testing
    class_data = train_test_split(X, y, test_size=.25)

    # init the data frame
    clfs = pd.DataFrame(index=['obj', 'grid'])

    # then add tests
    clfs['knn'] = pd.Series([KNeighborsClassifier(),
                            {'n_neighbors': np.arange(25) + 1,
                             'weights': ['uniform', 'distance']}],
                            index=clfs.index)
    clfs['lda'] = pd.Series([LDA(), {}], index=clfs.index)
    clfs['qda'] = pd.Series([QDA(), {}], index=clfs.index)
    clfs['svm'] = pd.Series([SVC(),
                            {'C': 2. ** np.arange(-6, 5),
                             'gamma': 2. ** np.arange(-6, 5),
                             'decision_function_shape': ['ovo', 'ovr']}],
                            index=clfs.index)
    sample_range = (2. ** np.arange(1, 11)).astype(int)
    clfs['random forest'] = pd.Series([RandomForestClassifier(),
                                       {'n_estimators': np.arange(1, 11) * 10,
                                        'min_samples_split': sample_range}],
                                      index=clfs.index)
    clfs['AdaBoost'] = pd.Series([AdaBoostClassifier(),
                                  {'n_estimators': np.arange(1, 11) * 10,
                                   'learning_rate': 10. ** np.arange(-4, 1)}],
                                 index=clfs.index)

    # analyses
    results = analysis(class_data, clfs)
#    with open('tables.txt') as f:
#        f.write(results)
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
