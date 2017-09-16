from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier, RandomizedLogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import RandomizedLasso

from sklearn.svm import LinearSVC
from sklearn import tree, linear_model
import numpy as np
import pandas as pd

def feature_selection_class(predictors, responses, test_predictors, selectFeatTech):
    if(selectFeatTech==0):
        t=int(predictors.shape[1]*0.40)
        t=10
        model = SelectKBest(chi2, k=t).fit(predictors.replace(-1,0), responses)
        #print model.scores_
        predictors_new = model.transform(predictors)
        predictors_test_new = model.transform(test_predictors)
        indices = model.get_support(indices=True)
    if(selectFeatTech==1):
        randomized_logistic = RandomizedLogisticRegression()
        model = randomized_logistic.fit(predictors, responses)
        predictors_new = model.transform(predictors)
        predictors_test_new = model.transform(test_predictors)
        indices = model.get_support(indices=True)

    column_names = predictors.columns[indices]
    predictors_new = pd.DataFrame(predictors_new, index=predictors.index, columns=column_names)
    predictors_test_new = pd.DataFrame(predictors_test_new, index=test_predictors.index, columns=column_names)
    return predictors_new, predictors_test_new

def apply_classification_model(X_train, y_train, X_test, y_test, selectModel):
    Result = {}
    Result['X_train'] = X_train
    Result['y_train'] = y_train
    Result['X_test'] = X_test
    Result['y_test'] = y_test

    if(selectModel== 'onevsrest'):
        print "OneVsRvest"
        classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(Result['X_train'], Result['y_train'])
    if(selectModel== 'linearsvc'):
        print "Linear SVC"
        classifier = LinearSVC(random_state=0).fit(Result['X_train'], Result['y_train'])
    if(selectModel== 'decisiontree'):
        print "Decision Tree"
        classifier = tree.DecisionTreeClassifier().fit(Result['X_train'], Result['y_train'])
    if(selectModel== 'nearestcentroid'):
        print "Nearest Centroid"
        classifier = NearestCentroid().fit(Result['X_train'], np.ravel(Result['y_train']))
    if(selectModel== 'sgd'):
        print "SGD Classifier"
        classifier = SGDClassifier(loss="hinge", penalty="l2").fit(Result['X_train'], np.ravel(Result['y_train']))
    if (selectModel == 'logisticregress'):
        print 'Logistic regression'
        classifier = LogisticRegressionCV().fit(Result['X_train'], np.ravel(Result['y_train']))

    predictions = classifier.predict(Result['X_test'])
    predictions_train = classifier.predict(Result['X_train'])
    Result['predictions'] = pd.DataFrame(predictions, index=X_test.index, columns=['predictions'])
    Result['model'] = classifier
    #Result['raw_model'] = pd.Series(classifier.coef_, index = X_train.columns)
    Result['predictions_train'] = pd.DataFrame(predictions_train, index=X_train.index, columns=['predictions'])
    return Result

def feature_selection_regression(predictors, responses, test_predictors, selectfeattech):
    if selectfeattech == 0:
        chk = int(predictors.shape[1] * 0.40)
        # have fixed the value of how many features are to be selected as of now.
        model = SelectKBest(f_regression, k=10)
        model = model.fit(predictors, responses)
        predictors_new = model.transform(predictors)
        predictors_test_new = model.transform(test_predictors)
        indices = model.get_support(indices=True)
        print "SelectKBest -> " + str(len(indices))

    if selectfeattech == 1:
        model = RandomizedLasso(alpha='aic', scaling=0.3, sample_fraction=0.60, n_resampling=200,
                                selection_threshold=0.15)
        model = model.fit(predictors, responses)
        predictors_new = model.transform(predictors)
        predictors_test_new = model.transform(test_predictors)
        indices = model.get_support(indices=True)
        print "Randomized Lasso -> " + str(len(indices))

    column_names = predictors.columns[indices]
    predictors_new = pd.DataFrame(predictors_new, index=predictors.index, columns=column_names)
    predictors_test_new = pd.DataFrame(predictors_test_new, index=test_predictors.index, columns=column_names)
    return predictors_new, predictors_test_new

def apply_regression_model(X_train, y_train, X_test, y_test, selectModel):
    Result = {}
    Result['X_train'] = X_train
    Result['y_train'] = y_train
    Result['X_test'] = X_test
    Result['y_test'] = y_test
    if (selectModel == 'linear'):
        print "Linear Regression"
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions_train = model.predict(X_train)
    if (selectModel == 'ridge'):
        print "Ridge Regression"
        model = linear_model.RidgeCV(alphas=(0.1, 0.1, 10))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions_train = model.predict(X_train)
    if (selectModel == 'lasso'):
        print "Lasso Regression"
        model = linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=(0.1, 0.1, 10))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions_train = model.predict(X_train)
    Result['predictions'] = pd.DataFrame(predictions, index=X_test.index, columns=['predictions'])
    Result['model'] = model
    Result['predictions_train'] = pd.DataFrame(predictions_train, index=X_train.index, columns=['predictions'])
    return Result


