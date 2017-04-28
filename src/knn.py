from __future__ import division

import csv
import random
import math
import operator
from datetime import datetime
import string
import heapq

import numpy as np
import nltk

tokenize = lambda doc: doc.lower().split(" ")

def idf(term, comments):
    comments = [tokenize(d) for d in comments]
    contains_term = map(lambda comm: term in comm, comments)
    if sum(contains_term) != 0:
        return 1 + math.log(len(comments)/(sum(contains_term)))
    else:
        return 1

def norm_tf(term, comment):
    return comment.count(term)/ len(comment)

def createFeatures(comments, dictionary):
    dataset = []
    idftable = {}
    print "#1: Entered Create Features- ", len(comments), len(dictionary)
    for word in dictionary:
        idftable[word] = idf(word, comments)

    print "#2: idftable created - ", len(idftable)

    for comment in comments:
        features = []
        for word in dictionary:
            features.append(norm_tf(word,comment) * idftable[word])
        dataset.append(features)
    
    print "#3: features created - ", dataset
    return dataset

def loadInputSet(dataset, split, trainInput=[], cvInput=[], trainIndex = [], cvIndex = []):
    random.seed(datetime.now())
    m,n = dataset.shape   
 
    for x in range(m):
        if random.random() < split:
            trainInput.append(dataset[x])
            trainIndex.append(x)
        else:
            cvInput.append(dataset[x])
            cvIndex.append(x)

def loadOutputSet(dataset, trainIndex, cvIndex, trainOutput=[], cvOutput=[]):

    trainOutput += list(dataset[i] for i in trainIndex)
    cvOutput += list(dataset[i] for i in cvIndex)

def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def cosineDistance(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def getNeighbors(trainInput, testInstance, k):
    distances = []

    for x in range(len(trainInput)):
        print "inner iteration #", x
        dist = euclideanDistance(testInstance, trainInput[x])
        distances.append((x, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def getResponse(neighbors, trainOutput):
    classVotes = {}

    for x in neighbors:
        response = trainOutput[x] 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(cvOutput, predictions):
    correct = 0
    for x in range(len(cvOutput)):
        if cvOutput[x] == predictions[x]:
            correct += 1
    return (correct/float(len(cvOutput))) * 100.0
    
def main():
    # prepare data
    accList = []

    split = 0.70
     
    NumIterations = 1
    FullPath = "/home/srikanth.pasumarthy/COMP551/COMP_551_P2"    

    with open(FullPath + '/csv_files/train_input_processed_2_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset1 = list(lines)

    with open(FullPath + '/csv_files/dictionary_frequency_2_grams.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)

    dataset2 = [[row[1] for row in dataset2[1:][:]]]
    dataset2 = [item for sublist in dataset2 for item in sublist]

    dataset1 = createFeatures([row[1] for row in dataset1[1:][:]], dataset2)
    dataset1 = np.array([[float(r) for r in row[1:]] for row in dataset1[1:][:]])
    print "#1:", dataset1.shape

    with open(FullPath + '/csv_files/test_input_vectorized.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset3 = list(lines)

    dataset3 = np.array([[float(r) for r in row[1:]] for row in dataset3[1:][:]])
    dataset3 = createFeatures([row[1] for row in dataset3[1:][:]], dataset2)

    print "#2:", dataset3.shape

    with open(FullPath + '/csv_files/train_output.csv', 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset2 = list(lines)
        dataset2 = np.array([[row[1] for row in dataset2[1:][:]]])
        dataset2 = np.transpose(dataset2)
    print "#3:", dataset2.shape
    
    #from sklearn.utils import shuffle
    #dataset1, dataset2 = shuffle(dataset1, dataset2, random_state=0)

    predictions = []

    ofile  = open('prediction_knn.csv', "wb")
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

    k = 3
    m,n = dataset3.shape
    for i in range(m):
        s = euclideanDistance(dataset1, dataset3[i][:])
        ind = heapq.nsmallest(k, range(len(s)), s.take)
        result = getResponse(ind, dataset2)

        row = []
        row.append(i)
        row.append(result)
        writer.writerow(row)

        predictions.append(result)
        print "#debug:", i, ":", s.shape, result

    print "#done: ", len(predictions)

    ofile.close()

    ''' 
    for iteration in range(NumIterations):

        trainInput=[]
        cvInput=[]
        trainIndex=[]
        cvIndex=[]
        trainOutput=[]
        cvOutput=[]

        loadInputSet(dataset1, split, trainInput, cvInput, trainIndex, cvIndex)
        loadOutputSet(dataset2, trainIndex, cvIndex, trainOutput, cvOutput)
        print "#0: length of data:", dataset1.shape
        
        print 'Size of Train set: ' + repr(len(trainIndex)), len(trainInput), len(trainOutput)
        print 'Size of Validation set: ' + repr(len(cvIndex)), len(cvInput), len(cvOutput)
        # generate predictions
        predictions=[]
        k += 1
        for x in range(len(cvIndex)):
            print "outer iteration #:", x
            s = pairwiseDistance(trainInput, cvInput[x])
            print "outer iteration #:", x, s.shape
            #neighbors = getNeighbors(trainInput, cvInput[x], k)
            #result = getResponse(neighbors, trainOutput)
            #predictions.append(result)
            #print('> predicted=' + repr(result) + ', actual=' + repr(validationSet[x][-1]))

        accuracy = getAccuracy(cvOutput, predictions)
        accList.append((k, accuracy))
        print('Accuracy: ' + repr(accuracy) + '%')
         
    
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(accList)))
    '''
       
main()
