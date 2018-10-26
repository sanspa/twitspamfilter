#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:08:52 2018

@author: budi
"""

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re
from substitusi import *


def regTokenize(text):
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    return words 
    
def displayfeature(tf_counts):
    counts = dict(Counter(tf_counts).most_common(20))
    labels, values = zip(*counts.items())    
    # sort your values in descending order
    indSort = np.argsort(values)[::-1]    
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]   
    indexes = np.arange(len(labels))    
    bar_width = 0.35   
    plt.bar(indexes, values)   
    plt.xticks(indexes + bar_width, labels,rotation='vertical')
    plt.show()


def praproses(text):
    text = sub_clean(text,pengganti) 
    text = wordfilter(text)
    return text
    

def createToken(text,gram=1):
    words = text.split()
    if gram > 1:
        word = []
        for i in range(len(words) - gram + 1):
            word += [' '.join(words[i:i + gram])]
        return word
    return words

def prosestext(text):
    text = sub_clean(text,pengganti)   
    stemfactory = StemmerFactory()
    stemmer = stemfactory.create_stemmer()
    text = stemmer.stem(text)
    #words = texts.split()
    return text

