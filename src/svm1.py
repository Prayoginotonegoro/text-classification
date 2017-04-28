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

def getErrorDist(cvOutput, predictions, label, lookupTable):
    s = {}

    indexes = [i for i,x in enumerate(cvOutput) if x == label]
    print "#debug1: ", len(indexes)
    import operator
    errors = operator.itemgetter(*indexes)(predictions)
    print "#debug2: ", len(errors)
    for x in lookupTable:
        s[x] = errors.count(x)
    
    return s

def main():
    DataSetLen = 164998
    Split = 0.80
    ''' 
    #####################  Unigram - TF IDF ##############################
    with open('../csv_files/train_input_processed_1_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset1 = list(lines)

    with open('../csv_files/dictionary_frequency_1_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = [[row[1] for row in dataset2[1:][:]]]
        dataset2 = [item for sublist in dataset2 for item in sublist]

    dataset1 = createFeatures_scikit([row[1] for row in dataset1[1:][:]], dataset2)

    with open('../csv_files/train_output.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = np.array([[row[1] for row in dataset2[1:][:]]])
        dataset2 = np.transpose(dataset2)

    lookupTable, dataset2 = np.unique(dataset2, return_inverse=True) 
    print "debug Len: ", dataset1.shape, dataset2.shape

    dataset1, dataset2 = shuffle(dataset1, dataset2)
    dataset1 = dataset1[:DataSetLen][:]
    dataset2 = dataset2[:DataSetLen][:]
    index = int(Split * DataSetLen)
     
    trainInput = dataset1[:index][:]
    trainOutput = dataset2[:index]
    cvInput = dataset1[index:][:]
    cvOutput = dataset2[index:]
        
    CValue = 1.0
    NumIterations = 1;
    
    for iteration in range(NumIterations):
    #clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', verbose = True, probability = True), n_jobs = 10)
        clf = OneVsRestClassifier(svm.LinearSVC(C= CValue), n_jobs = 10)
    #print "debug #0"
        clf.fit(trainInput, trainOutput)
    #print "debug #1: fit completed"
        score = clf.score(cvInput,cvOutput)
        print "debug #: Training Completed; Score = ", score, " ;C = ", CValue
        CValue *= 2
    ''' 
     
    ############################# TFIDF bigram ##################################

    
    with open('../csv_files/train_input_processed_2_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset1 = list(lines)

    with open('../csv_files/dictionary_frequency_2_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = [[row[1] for row in dataset2[1:][:]]]
        dataset2 = [item for sublist in dataset2 for item in sublist]

    dataset1 = createFeatures_scikit([row[1] for row in dataset1[1:][:]], dataset2)

    with open('../csv_files/train_output.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = np.array([[row[1] for row in dataset2[1:][:]]])
        dataset2 = np.transpose(dataset2)

    lookupTable, dataset2 = np.unique(dataset2, return_inverse=True) 
    print "debug Len1: ", dataset1.shape, dataset2.shape, lookupTable

    dataset1, dataset2 = shuffle(dataset1, dataset2)
    dataset1 = dataset1[:DataSetLen][:]
    dataset2 = dataset2[:DataSetLen][:]
    index = int(Split * DataSetLen)

    trainInput = dataset1[:index][:]
    trainOutput = dataset2[:index]
    cvInput = dataset1[index:][:]
    cvOutput = dataset2[index:]
        
    CValue = 1000
    NumIterations = 1;

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    for iteration in range(NumIterations):
    #clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', verbose = True, probability = True), n_jobs = 10)
        clf = OneVsRestClassifier(svm.LinearSVC(C= CValue), n_jobs = 10)
    #print "debug #0"
        clf.fit(trainInput, trainOutput)
        predictions = clf.predict(cvInput)
    #print "debug #1: fit completed"
        score = clf.score(cvInput,cvOutput)
        cfmatrix = confusion_matrix(cvOutput, predictions)
        cfreport = classification_report(cvOutput, predictions, target_names = lookupTable)
        
        errorDist1 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'news', lookupTable)
        errorDist2 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'worldnews', lookupTable)
        errorDist3 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'politics', lookupTable)
        errorDist4 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'hockey', lookupTable)
        errorDist5 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'movies', lookupTable)
        errorDist6 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'nfl', lookupTable)
        errorDist7 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'soccer', lookupTable)
        errorDist8 = getErrorDist(lookupTable[cvOutput], lookupTable[predictions], 'nba', lookupTable)

        print "debug #: Training Completed; "
        print "Score = ", score,
        print "Confusion Matrix: ", cfmatrix
        print "CF Report: ", cfreport
        print "Error Distributions:" 
        print errorDist1
        print errorDist2
        print errorDist3
        print errorDist4
        print errorDist5
        print errorDist6
        print errorDist7
        print errorDist8

        CValue *= 2

    ''' 
    ############################# Word2Vec TFIDF #########################
    with open('../csv_files/train_input_idf_vectorized.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset1 = list(lines)
        dataset1 = np.array([[float(r) for r in row[1:]] for row in dataset1[1:][:]])

    with open('../csv_files/train_output.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = np.array([[row[1] for row in dataset2[1:][:]]])
        dataset2 = np.transpose(dataset2)
    
    lookupTable, dataset2 = np.unique(dataset2, return_inverse=True) 
    print "debug Len2: ", dataset1.shape, dataset2.shape

    dataset1, dataset2 = shuffle(dataset1, dataset2)
    dataset1 = dataset1[:DataSetLen][:]
    dataset2 = dataset2[:DataSetLen][:]
    index = int(Split * DataSetLen)

    trainInput = dataset1[:index][:]
    trainOutput = dataset2[:index]
    cvInput = dataset1[index:][:]
    cvOutput = dataset2[index:]
        
    CValue = 1.0
    NumIterations = 1;

    for iteration in range(NumIterations):
    #clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', verbose = True, probability = True), n_jobs = 10)
        clf = OneVsRestClassifier(svm.LinearSVC(C= CValue), n_jobs = 10)
    #print "debug #0"
        clf.fit(trainInput, trainOutput)
    #print "debug #1: fit completed"
        score = clf.score(cvInput,cvOutput)
        print "debug #: Training Completed; Score = ", score, " ;C = ", CValue
        CValue *= 2

    #clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', verbose = True), n_jobs = 10)
    #print "debug #2.0"
    #clf.fit(trainInput, trainOutput)
    #print "debug #2.1: fit completed"
    #score = clf.score(cvInput,cvOutput)
    #print "debug #2.2: Training Completed; Score = ", score
     
    ############################# Word2Vec Mean #############################

    with open('../csv_files/train_input_mean_vectorized.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset1 = list(lines)
        dataset1 = np.array([[float(r) for r in row[1:]] for row in dataset1[1:][:]])

    with open('../csv_files/train_output.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = np.array([[row[1] for row in dataset2[1:][:]]])
        dataset2 = np.transpose(dataset2)
    

    lookupTable, dataset2 = np.unique(dataset2, return_inverse=True) 
    print "debug Len3: ", dataset1.shape, dataset2.shape

    dataset1, dataset2 = shuffle(dataset1, dataset2)
    dataset1 = dataset1[:DataSetLen][:]
    dataset2 = dataset2[:DataSetLen][:]
    index = int(Split * DataSetLen)

    trainInput = dataset1[:index][:]
    trainOutput = dataset2[:index]
    cvInput = dataset1[index:][:]
    cvOutput = dataset2[index:]
        
    CValue = 1.0
    NumIterations = 1;

    for iteration in range(NumIterations):
    #clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', verbose = True, probability = True), n_jobs = 10)
        clf = OneVsRestClassifier(svm.LinearSVC(C= CValue), n_jobs = 10)
    #print "debug #0"
        clf.fit(trainInput, trainOutput)
    #print "debug #1: fit completed"
        score = clf.score(cvInput,cvOutput)
        print "debug #: Training Completed; Score = ", score, " ;C = ", CValue
        CValue *= 2

    #clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', verbose = True), n_jobs = 10)
    #print "debug #3.0"
    #clf.fit(trainInput, trainOutput)
    #print "debug #3.1: fit completed"
    #score = clf.score(cvInput,cvOutput)
    #print "debug #3.2: Training Completed; Score = ", score
    ''' 
main()
