#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:28:47 2017

@author: al
"""

import pandas as pd
#import numpy as np
import nltk
 
# get the csv
df = pd.read_csv("../csv_files/train_input_processed.csv")

tokens = []

#get all the words that appear in the corpus
for index,row in df.iterrows():
    print index
    tokens.extend(row['conversation'].split(" "))
    
    

project_2_dictionary = []


# find the frequency distribution of these tokens
freq_dist = nltk.FreqDist(tokens)
# filter out the rare words
most_common = freq_dist.most_common(30000)

words = [item[0] for item in most_common]
counts = [item[1] for item in most_common]
      
                                        
out_df = pd.DataFrame({"word":words})

outfile = "../csv_files/dictionary_distribution.csv"

out_df.to_csv(outfile, index=False)

