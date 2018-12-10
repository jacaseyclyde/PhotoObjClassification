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

import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, make_scorer
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
warnings.filterwarnings("ignore")


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


def load_data(pos=False, op=False):
    # There is the possibility of this containing duplicate objects,
    # but because each row represents a seperate observation, it's ok
    # and we don't need to filter for them
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'data', 'sdss.csv'))

    # Get rid of features that don't represent physical data or
    # something directly derived from it (i.e., not a fit).
    # Commented features are currently considered as useful for
    # classification
    drop = ['specObjID', 'mjd', 'plate', 'tile', 'fiberID', 'z', 'zErr',
            'zWarning', 'htmID', 'survey', 'programname', 'legacy_target1',
            'legacy_target2', 'plateID', 'sourceType', 'targetObjID', 'objID',
            'skyVersion', 'run', 'rerun', 'camcol', 'field', 'obj', 'flags',
            'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i',
            'modelMag_z', 'modelMagErr_u', 'modelMagErr_g', 'modelMagErr_r',
            'modelMagErr_i', 'modelMagErr_z', 'cModelMag_u', 'cModelMag_g',
            'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 'cModelMagErr_u',
            'cModelMagErr_g', 'cModelMagErr_r', 'cModelMagErr_i',
            'cModelMagErr_z', 'mRrCc_r', 'mRrCcErr_r', 'fieldID', 'dered_u',
            'dered_g', 'dered_r', 'dered_i', 'dered_z']
    if not pos:
        drop += ['ra', 'dec', 'cx', 'cy', 'cz', 'photoRa', 'photoDec']

    if not op:
        drop += ['sciencePrimary', 'legacyPrimary',
                 'seguePrimary', 'segue1Primary', 'segue2Primary',
                 'bossPrimary', 'sdssPrimary', 'special_target1',
                 'special_target2', 'segue1_target1', 'segue1_target2',
                 'segue2_target1', 'segue2_target2', 'boss_target1',
                 'ancillary_target1', 'ancillary_target2', 'mode', 'nChild',
                 'score', 'resolveStatus']
    data = data.drop(labels=drop,
                     axis=1)

    return data


def grid_search_optimizer(data, clf, params, var=None, cv=5,
                          scorer=None):
    X_train, X_test, y_train, y_test = data

    if var < 1:
        pca = PCA(n_components=var)
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    else:
        pca = PCA()
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf_cv = GridSearchCV(clf, params, cv=cv, n_jobs=-1, scoring=scorer)
    clf_cv.fit(X_train, y_train)

    err = 1 - clf_cv.score(X_test, y_test)
    return err, clf_cv.best_params_, clf_cv.cv_results_


def count_experiments(clfs):
    n_exps = np.ones(len(clfs.columns), dtype=np.int16)
    for i, clf in enumerate(clfs):
        param_grid = clfs[clf]['grid']

        for param in param_grid:
            n_exps[i] *= len(param_grid[param])

    return n_exps


def custom_score(y, y_pred, pbar=None):
    matr = confusion_matrix(y, y_pred)
    if pbar is not None:
        pbar.update(1)

    return np.sum(np.diagonal(matr)) / np.sum(matr)


def analysis(data, tests, prefix=None):
    n_exps = count_experiments(tests)
    cv = 5

    var_max = 1.
    var_min = .75
    variances = np.linspace(var_min, var_max,
                            num=6)
    variances = np.around(variances, 2)

    ind = pd.MultiIndex.from_product([(100 * variances).astype(int),
                                      ['err', 'best_params', 'all_params']],
                                     names=['pca', 'results'])
    results = pd.DataFrame(index=ind, columns=tests.columns)

    n_tests = cv * sum(n_exps) * len(variances)
    with tqdm(total=n_tests, file=sys.stdout) as pbar:
        # TODO: Make this work
#        scorer = make_scorer(custom_score, pbar=pbar)
        for i, test in enumerate(tests):
            clf = tests[test]['obj']
            params = tests[test]['grid']
            for var in variances:  # variance retained
                pbar.set_description("clf: {0}, "
                                     "pca: {1}".format(test,
                                                       var))

                err, par_best, par_all = grid_search_optimizer(data, clf,
                                                               params,
                                                               var=var,
                                                               scorer=None)

                results[test][int(100 * var), 'err'] = round(err, 4)
                results[test][int(100 * var), 'best_params'] = par_best
                results[test][int(100 * var), 'all_params'] = par_all

                pbar.update(cv * n_exps[i])

    errs = results.loc[(results.index.get_level_values('results')
                        == 'err')].reset_index(level=1, drop=True)

    pbest = results.loc[(results.index.get_level_values('results')
                        == 'best_params')].reset_index(level=1, drop=True)

    pall = results.loc[(results.index.get_level_values('results')
                        == 'all_params')].reset_index(level=1, drop=True)

    results.to_csv(os.path.join(os.path.dirname(__file__),
                                'out', '{0}_results.csv'.format(prefix)),
                   index_label=results.index.names)

    errs.to_csv(os.path.join(os.path.dirname(__file__),
                             'out', '{0}_errs.csv'.format(prefix)),
                index_label=results.index.names)

    pbest.to_csv(os.path.join(os.path.dirname(__file__),
                              'out', '{0}_pbest.csv'.format(prefix)),
                 index_label=results.index.names)

    pall.to_csv(os.path.join(os.path.dirname(__file__),
                             'out', '{0}_pall.csv'.format(prefix)),
                index_label=results.index.names)

    return results, errs, pbest, pall


def plot_errors(errs, prefix=None):
    indicies = errs.index.values.astype(float)

    plt.figure()
    marker = itertools.cycle(('o', '^', '+'))

    for clf in errs:
        plt.plot(indicies, errs[clf].values,
                 label=clf,
                 marker=next(marker))

    plt.ylabel("Error Rate")
    plt.xlabel("PCA % Variance Retained")

    lgd = plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title("Error Rates vs. PCA Variance")

    save_path = os.path.join(os.path.dirname(__file__),
                             'out/{0}_errors.pdf'.format(prefix))
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def main():
    prefixes = ["mag", "mag_op", "mag_pos", "mag_op_pos"]
    for prefix in prefixes:
        pos = prefix == "mag_pos" or prefix == "mag_op_pos"
        op = prefix == "mag_op" or prefix == "mag_op_pos"
        data = load_data(pos=pos, op=op)

        # Our two types of labels
        classes = data.pop('class')
        subclasses = data.pop('subClass')

        # initialize dicts relating integer class values to their human
        # friendly string values
        _class_dict_init(classes)
        _subclass_dict_init(subclasses)

        # init the data frame
        clfs = pd.DataFrame(index=['obj', 'grid'])

        # then add tests
        clfs['kNN'] = pd.Series([KNeighborsClassifier(),
                                {'n_neighbors': np.arange(25) + 1,
                                 'weights': ['uniform', 'distance']}],
                                index=clfs.index)
        clfs['LDA'] = pd.Series([LDA(), {}], index=clfs.index)
        # QDA Performs extremely poorly without LDA, leaving commented for now
    #    clfs['QDA'] = pd.Series([QDA(), {}], index=clfs.index)
        clfs['SVM'] = pd.Series([SVC(),
                                {'C': 2. ** np.arange(-6, 5),
                                 'gamma': 2. ** np.arange(-6, 5),
                                 'decision_function_shape': ['ovo', 'ovr']
                                 }],
                                index=clfs.index)
        sample_range = (2. ** np.arange(1, 11)).astype(int)
        clfs['Random Forest'] = pd.Series([RandomForestClassifier(),
                                           {'n_estimators': np.arange(1, 11) * 10,
                                            'min_samples_split': sample_range
                                            }],
                                          index=clfs.index)
        clfs['AdaBoost'] = pd.Series([AdaBoostClassifier(),
                                      {'n_estimators': np.arange(1, 11) * 10,
                                       'learning_rate': 10. ** np.arange(-4, 1)
                                       }],
                                     index=clfs.index)
        clfs['MLP'] = pd.Series([MLPClassifier(),
                                 {'hidden_layer_sizes': [(), (2 ** 3, ),
                                                         (2 ** 4, ), (2 ** 5, ),
                                                         (2 ** 6, ),
                                                         (2 ** 3, 2 ** 3),
                                                         (2 ** 4, 2 ** 3),
                                                         (2 ** 5, 2 ** 3),
                                                         (2 ** 6, 2 ** 3),
                                                         (2 ** 4, 2 ** 4),
                                                         (2 ** 5, 2 ** 4),
                                                         (2 ** 6, 2 ** 4),
                                                         (2 ** 5, 2 ** 5),
                                                         (2 ** 6, 2 ** 5),
                                                         (2 ** 6, 2 ** 6)],
                                  'activation': ['identity', 'logistic',
                                                 'tanh', 'relu'],
                                  'learning_rate_init':
                                      10. ** np.arange(-4, 2)
                                  }],
                                index=clfs.index)

        # analyses
        # classes
        X = data.values
        y_cls = np.vectorize(CLASS_DICT.get)(classes)

        # split data for training/testing
        class_data = train_test_split(X, y_cls, test_size=500, train_size=1500,
                                      random_state=1234567890)

        cls_results, cls_errs, cls_pbest, cls_pall = analysis(class_data, clfs,
                                                              prefix=prefix)
        save_path = os.path.join(os.path.dirname(__file__),
                                 'out/{0}_results.tex'.format(prefix))
        cls_results.to_latex(save_path)

        plot_errors(cls_errs, prefix=prefix)
        print(np.min(cls_errs.values))
        print(np.min(cls_errs))

    return class_data


if __name__ == '__main__':
    figure = {'figsize': (12, 12)}
    mpl.rc('figure', **figure)

    font = {'size': 24}
    mpl.rc('font', **font)

    directory = os.path.join(os.path.dirname(__file__), 'out')
    if not os.path.exists(directory):
        os.makedirs(directory)
    class_data = main()
