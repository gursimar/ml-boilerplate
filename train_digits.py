import os, glob
import pandas as pd
from mlengines import *
from mlresults import *

def convertToIntArray(content):
    arr = []
    for ch in content:
        if ch != '\r' and ch!='\n':
            arr.append(int(ch))
    return arr

def readFiles():
    # Initialize lists here
    names = []
    contents = []
    features = []
    labels = []

    for file in glob.glob("*.txt"):
        content = open(file,'r').read()
        arr = convertToIntArray(content)
        features.append(arr)
        contents.append(content)
        name = file.split('.')[0]
        label = int(name.split('_')[0])
        names.append(name)
        labels.append(label)

    return {
        #'names': names,
        #'contents': contents,
        'features': features,
        'labels': labels
    }

def classify(X, y):
    print 'simar'


if __name__ == '__main__':

    #Notes:
    # Image size - 32x32 = 1024

    folder_train = '../datasets/digits-uav/trainingDigits/'
    folder_test = '../testDigits/'
    os.chdir(folder_train)
    train_data = readFiles()
    os.chdir(folder_test)
    test_data = readFiles()

    print 'Size of train data ' + str(len(train_data['labels']))
    print 'Size of test data ' + str(len(test_data['labels']))


    train_mlrs = []
    test_mlrs = []
    for model in ['onevsrest', 'linearsvc' ,'decisiontree' ,'nearestcentroid', 'sgd', 'logisticregress']:
        temp = apply_classification_model(pd.DataFrame(train_data['features']), train_data['labels'],
                               pd.DataFrame(test_data['features']), test_data['labels'], model)

        train_mlr = ClassificationResult(train_data['labels'], temp['predictions_train'], model + '_train')
        test_mlr = ClassificationResult(test_data['labels'], temp['predictions'], model + '_test')
        train_mlrs.append(train_mlr)
        test_mlrs.append(test_mlr)

        print '=====' + model +'====='
        print '==Train=='
        train_mlr.print_aggregate_stats()
        print ''
        print '==Test=='
        test_mlr.print_aggregate_stats()
        print ''
        print ''

    # Dump raw data
    os.chdir('../../../ml-boilerplate/results')
    dumpRawData(train_mlrs,'rawdata_train.csv')
    dumpRawData(test_mlrs,'rawdata_test.csv')

    # Dump Stats data
    dumpStatsData(train_mlrs,'stats_train.csv')
    dumpStatsData(test_mlrs,'stats_test.csv')

    # train a classifier without any dimention reduction (just to establish baseline)