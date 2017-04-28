#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:50:51 2017

@author: al
"""

import pandas as pd
import numpy as np
from copy import copy

test_train = "test"

most_common_count = 2000
outfile = "../csv_files/{0}_vectors_{1}.csv".format(test_train,most_common_count)

# get the conversations
df_train = pd.read_csv("../csv_files/{}_input_processed.csv".format(test_train))

# the dataframe will be created from this
output = []

# get the dictionary
df_dictionary = pd.read_csv("../csv_files/dictionary_distribution.csv").head(most_common_count)

# make the table that will hold the vectors
#output = pd.DataFrame(pd.np.empty((len(df_train), most_common_count)) * 0) 

template = {}

## build up the template from the words in the project dictionary
for index, row in df_dictionary.iterrows():
    template[row['word']] = 0

# now go through each of the conversations
for index, row in df_train.iterrows():
    print index
#    # copy the template to reset each word's count at 0
    conversation_count = copy(template)
    
    # get the list of words that appear in this conversation
    word_list = row['conversation'].split(" ")
    
    # go through each of the words and count how many times they show up
    for word in word_list:
        if any(df_dictionary.word == word):
            conversation_count[word] += 1
#            output.ix[index,df_dictionary[df_dictionary['word'] == word].index[0]] += 1
    
    # add a copy of of the count to the output
    output.append(copy(conversation_count))
    
    # save the vector, reducing memory usage
    if (index%2000 == 0) and (index > 0):   
        df = pd.DataFrame(output)         
        if index == 2000:       
            print "writing initial file " + outfile
            df.to_csv(outfile,index = False)
        else:
            print "appending to file " + outfile
            with open(outfile, 'a') as f:
                df.to_csv(f, header=False, index=False)
        output = []
    
print "appending to file " + outfile
with open(outfile, 'a') as f:
    df.to_csv(f, header=False, index=False)
## create a dataframe from the conversations counts
#df = pd.DataFrame(output)
#
##save the dataframe in a csv
#df.to_csv("../csv_files/word_counts.csv")