import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

def parseExcel(excel_file_name, type):
    # Sheets -> sid_lang
    # Compulsory -> lcs, sources
    # Optional -> tc, labels
    # variable -> rest of fields are treated as features
    excel_file = pd.ExcelFile(excel_file_name)
    sheet_names = excel_file.sheet_names
    CGs = {}
    for sheet_name in sheet_names:
        try:
            model_name, model_type = sheet_name.split(' | ')
            sid, lang = model_name.split('_')
        except:
            continue
        DF = excel_file.parse(sheet_name)
    return CGs


def dumpRawData(MLRs):
    if (type(MLRs) == 'dict'):
        MLRsList = MLRs.values()

    for MLR in MLRs:
        pass
    pass

def dumpResults(MLRs):
    pass

class MLResultFeatures:
    def __init__(self, predictions):
        self.predictions = predictions

class MLResult:
    def __init__(self, predictions):
        self.predictions = predictions

class SupervisedMLResult(MLResult):
    def __init__(self, labels, predictions):
        MLResult.__init__(self, predictions)
        self.labels = labels

    def getCorr(self):
        corr = pearsonr(self.predictions.values, self.labels.values)[0]
        return corr

class RegressionResult(SupervisedMLResult):
    def __init__(self, labels, predictions):
        SupervisedMLResult.__init__(self, labels, predictions)

    def discretizePredictionsRound(self):
        data = self.predictions
        data = np.round(data)
        data[data > 3] = 3
        #data[data < 1] = 1
        data[data < 0] = 0
        CR = ClassificationResult(data, self.labels)
        return CR


class ClassificationResult(SupervisedMLResult):
    def __init__(self, labels, predictions):
        SupervisedMLResult.__init__(self, labels, predictions)

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations known to be in
    # group i but predicted to be in group j.
    def getConfusionMat(self):
        mat = confusion_matrix(self.labels, self.predictions, )
        return mat

    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def getConfusionMatNorm(self):
        mat_norm = self.getConfusionMat().astype('float')/self.getConfusionMat().sum(axis=1)[:,np.newaxis]
        return mat_norm

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    def getMCR(self):
        accuracy = accuracy_score(self.labels, self.predictions)
        mcr = 1-accuracy
        return mcr

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    def getPrecision(self):
        precision = precision_score(self.labels, self.predictions)
        return precision

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    def getRecall(self):
        recall = recall_score(self.labels, self.predictions)
        return recall

    ''' Computes the F1 score of the positive class.
     In case of multiclass its the weighted avg of f1 score of each class'''
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    def getF1Score(self):
        f1_scor = f1_score(self.labels, self.predictions, average=None)
        return f1_scor
    pass
