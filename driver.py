import os, glob
import pandas as pd

if __name__ == '__main__':
    folder = 'results/freespeech/train'
    os.chdir(folder)

    # Initialize lists here
    names = []

    for file in glob.glob("input_*"):
        print file

        # read a file or a csv
        contents = open(file, 'r').read()
        data = pd.DataFrame.from_csv(file, index_col='User ID')

        for index, row in data.iterrows():
            # code to fetch data from csv and do operations
            print row
            name = row['Name']

            # maintain results of various rows
            names.append(name)

        results = pd.DataFrame({
            'name': names
        }, columns = ['name'])

    print len(results)

    # Dump average stats
    avgs = results.groupby(['type']).mean()
    counts = results.groupby(['type']).count()
    counts = counts[['name']]
    counts.rename(columns={'name': 'Counts'}, inplace=True)
    avgs = pd.concat([avgs, counts], axis=1, join='inner')
    medians = results.groupby(['type']).median()
    medians = pd.concat([medians, counts], axis=1, join='inner')
    maxs = results.groupby(['type']).max()
    maxs = pd.concat([maxs, counts], axis=1, join='inner')
    mins = results.groupby(['type']).min()
    mins = pd.concat([mins, counts], axis=1, join='inner')


    results.to_csv('rawData.csv')
    avgs.to_csv('avg.csv')
    medians.to_csv('median.csv')
    maxs.to_csv('max.csv')
    mins.to_csv('min.csv')
