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


def dumpRawData(MLRs, file_name):
    if (type(MLRs) == 'dict'):
        MLRsList = MLRs.values()

    # this assumes that all objects has same label set
    # so MLRs are result of same dataset with different techniques
    data_dict = {}
    data_dict['labels'] = MLRs[0].labels

    for MLR in MLRs:
        # assumes that its supervised ML object, can be later expanded
        # writerRaw = pd.ExcelWriter(file_name + '.csv');
        #data.to_excel(writerRaw, MLR.object_name) if you want in separate sheets
        data_dict[MLR.object_name] = MLR.predictions
    data = pd.DataFrame(data_dict)
    data.to_csv(file_name)

def dumpStatsData(MLRs, file_name):
    if (type(MLRs) == 'dict'):
        MLRsList = MLRs.values()
    if not isinstance(MLRs[0], ClassificationResult):
        print 'Does not support the object type' + str(type(MLRs[0]))

    all_results = []
    for MLR in MLRs:
        all_results.append(MLR.result_dict())
    pd.DataFrame(all_results).to_csv(file_name)
    pass

class MLResultFeatures:
    def __init__(self, predictions):
        self.predictions = predictions

class MLResult:
    def __init__(self, predictions, object_name):
        self.predictions = predictions
        self.object_name = object_name

class SupervisedMLResult(MLResult):
    def __init__(self, labels, predictions, object_name):
        MLResult.__init__(self, predictions, object_name)
        self.labels = labels

    def getCorr(self):
        corr = pearsonr(self.predictions, self.labels)[0]
        return corr

class RegressionResult(SupervisedMLResult):
    def __init__(self, labels, predictions, object_name):
        SupervisedMLResult.__init__(self, labels, predictions, object_name)

    def discretizePredictionsRound(self):
        data = self.predictions
        data = np.round(data)
        data[data > 3] = 3
        #data[data < 1] = 1
        data[data < 0] = 0
        CR = ClassificationResult(data, self.labels)
        return CR

class ClassificationResult(SupervisedMLResult):
    def __init__(self, labels, predictions, object_name):
        SupervisedMLResult.__init__(self, labels, predictions, object_name)

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
        precision = precision_score(self.labels, self.predictions, average = None)
        return precision

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    def getRecall(self):
        recall = recall_score(self.labels, self.predictions, average = None)
        return recall

    ''' Computes the F1 score of the positive class.
     In case of multiclass its the weighted avg of f1 score of each class'''
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    def getF1Score(self):
        f1_scor = f1_score(self.labels, self.predictions, average=None)
        return f1_scor

    # THESE FUNCTION NEEDS TO BE IMPLEMENTED IN ALL FUNCTIONS ABOVE
    def result_dict(self):
        result = {
            'name': self.object_name,
            'count': len(self.predictions),
            'corr': self.getCorr(),
            'mcr': self.getMCR(),
            'f1': self.getF1Score(),
            'precision': self.getPrecision(),
            'recall': self.getRecall(),
            'conf_mat': self.getConfusionMat(),
            'conf_mat_norm': self.getConfusionMatNorm()
        }
        return result

    def print_detailed_stats(self):
        try:
            print 'Count - ' + str(len(self.predictions))
            print 'MCR - ' + str(self.getMCR())
            print 'F1 - ' + str(self.getF1Score())
            print 'Precision - ' + str(self.getPrecision())
            print 'Recall - ' + str(self.getRecall())
            print 'Corr - ' + str(self.getCorr())
            print 'Confusion Mat'
            print self.getConfusionMat()
            print 'Confusion Mat Normalized'
            print self.getConfusionMatNorm()
        except:
            print 'mlresults: Some stats missing due to an error'
            pass

    def print_aggregate_stats(self):
        try:
            print 'Count - ' + str(len(self.predictions))
            print 'MCR - ' + str(self.getMCR())
            print 'F1 - ' + str(self.getF1Score())
            print 'Precision - ' + str(self.getPrecision())
            print 'Recall - ' + str(self.getRecall())
            #print 'Corr - ' + str(self.getCorr())
        except:
            print 'mlresults: Some stats missing due to an error'
            pass
    pass
