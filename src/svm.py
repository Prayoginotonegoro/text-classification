from __future__ import division

import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.utils import shuffle

import csv

tokenize = lambda doc: doc.lower().split(" ")

def createFeatures_scikit(comments,dictionary):
    from sklearn.feature_extraction.text import TfidfVectorizer
    sklearn_tfidf = TfidfVectorizer(min_df=0,sublinear_tf=True, dtype=np.int8, vocabulary = dictionary, tokenizer=tokenize)
    return sklearn_tfidf.fit_transform(comments)

def getAccuracy(cvOutput, predictions):
    correct = 0
    for x in range(len(cvOutput)):
        if cvOutput[x] == predictions[x]:
            correct += 1
    return (correct/float(len(cvOutput))) * 100.0

def main():
    with open('../csv_files/train_input_processed_2_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset1 = list(lines)

    with open('../csv_files/dictionary_frequency_2_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)

    dataset2 = [[row[1] for row in dataset2[1:][:]]]
    dataset2 = [item for sublist in dataset2 for item in sublist]

    dataset1 = createFeatures_scikit([row[1] for row in dataset1[1:][:]], dataset2)
    print "#1:", dataset1.shape

    with open('../csv_files/test_input_processed_2_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset3 = list(lines)

    dataset3 = createFeatures_scikit([row[1] for row in dataset3[1:][:]], dataset2)

    with open('../csv_files/train_output.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = np.array([[row[1] for row in dataset2[1:][:]]])
        dataset2 = np.transpose(dataset2)

    print "#2:", dataset2.shape, dataset3.shape

    lookupTable, dataset2 = np.unique(dataset2, return_inverse=True) 
    print "debug #0: ", dataset1.shape, dataset2.shape

    dataset1, dataset2 = shuffle(dataset1, dataset2)
    '''
    dataset1 = dataset1[:30000][:]
    dataset2 = dataset2[:30000][:]
    m,n = dataset1.shape
    index = int(0.90 * m)
    NumIterations = 1
    k = 1

    for iteration in range(NumIterations):
        trainInput = dataset1[:index][:]
        trainOutput = dataset2[:index]
        cvInput = dataset1[index:][:]
        cvOutput = dataset2[index:]
        
        clf = OneVsRestClassifier(svm.SVC(verbose = True, probability=True), n_jobs = 10)
        print "debug0"
        clf.fit(trainInput, trainOutput)
        print "debug1: fit completed"
        predictions = clf.predict(cvInput)
        score = clf.score(cvInput,cvOutput)
        print "debug #2: Training Completed; Score = ", score
        accuracy = getAccuracy(cvOutput, predictions)
        print('k = ' + str(k) + ' Accuracy = ' + str(accuracy) + '% score = ' + str(score))
        #accList.append((k,accuracy))

    #print('\n'.join('{}: {}'.format(*k) for k in enumerate(accList)))
    '''
    #clf = OneVsRestClassifier(svm.SVC(verbose = True, probability=True), n_jobs = 15)
    #n_estimators = 10
    #clf = OneVsRestClassifier(BaggingClassifier(svm.SVC(verbose = True, probability=True), n_estimators=n_estimators, n_jobs = 15), n_jobs = 15)
   
    CValue = 1000
    clf = OneVsRestClassifier(svm.LinearSVC(C= CValue), n_jobs = 10)
    clf.fit(dataset1, dataset2)
    predictions = clf.predict(dataset3)
    predictions = lookupTable[predictions]

    ofile  = open('prediction_svm_dict.csv', "wb")
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

    m,n = dataset3.shape

    row = []
    row.append('id')
    row.append('category')
    writer.writerow(row)

    for i in range(m):
        row = []
        row.append(i)
        row.append(predictions[i])
        writer.writerow(row)

    print "#done"

    ofile.close()
    
#clf = svm.SVC(kernel='linear', C = 1.0)
#clf.fit(X,y)
main()
