#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:19:11 2017

@author: al
"""
import numpy as np
import re
import gensim
import pandas as pd

def get_class_vectors(classes):
    class_vectors = {}

    for doc_class in classes:
        try:
            if doc_class == 'movies':
                class_vectors[doc_class] = conversation_to_vector(['movie'])
            else:
                class_vectors[doc_class] = conversation_to_vector([doc_class])
        except:
            print "no vector for ", doc_class
            
    return class_vectors
    

def conversation_to_vector(conversation_list):
    count = 0
    test_vec= [0]*200
    for word in conversation_list:
        try:
            test_vec += model[word]
            count+= 1
        except:
            pass
    return test_vec
        

def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    
def get_accuracy(inputs,label_options,result_labels):
    i = 0
    score = 0
    for conversation in inputs:
        print i
        conv_vec = conversation_to_vector(conversation)
        max_similarity = 0
        for label in label_options:
            similarity = cosine_similarity(label_options[label],conv_vec)
            if similarity > max_similarity:
                max_similarity = similarity
                best = label
        if best == result_labels[i] :
            score += 1
        i += 1
        print float(score)/i
        
    return float(score)/i

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open("../csv_files/{}.csv".format(self.fname)):
            # return a generator, removing the id and blank space that
            # arrives from splitting
            if re.split("\W+",line.lower())[0] != 'id':
                yield re.split("\W+",line.lower())[1:-1]
            

if __name__ == "__main__":
    model = gensim.models.Word2Vec.load('project2_word2vec_model_2_grams')
    class_df = pd.read_csv('../csv_files/train_output.csv')
    classes = list(class_df.category.unique())
    class_vectors = get_class_vectors(classes)
    
            
    # test on training data
    
    sentences = MySentences('train_input_processed_2_grams')
    train_classes = list(class_df['category'])
    
    acc = get_accuracy(sentences,class_vectors,train_classes)
        
    
        