#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 01:21:54 2017

@author: al
"""

import gensim as gs
import numpy as np
import re

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open("../csv_files/{}.csv".format(self.fname)):
            # return a generator, removing the id and blank space that
            # arrives from splitting
            if re.split("\W+",line.lower())[0] != 'id':
                yield re.split("\W+",line.lower())[1:-1]

n_grams = 2

sentences = MySentences('train_input_processed_2_grams'.format())
model = gs.models.Word2Vec(iter=5,size=200,workers=4,min_count=10)
model.build_vocab(sentences)
sentences = MySentences('test_input_processed_2_grams')
model.train(sentences)
model.init_sims(replace=True)
model.save('project2_word2vec_model_2_grams')