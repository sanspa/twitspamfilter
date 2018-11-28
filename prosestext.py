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
import csv
import re
import html
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


def clean_text(text):
    text = html.unescape(text)
    text = text.replace('_',' ')
    t2 = ' '.join(text.split())
    return t2

def konversi(text,pengganti):
    text = text.lower()
    for k,v in pengganti.items():
        text=re.sub(k,v,text)
    return text     

def createToken(text,gram=1):
    words = text.split()
    if gram > 1:
        word = []
        for i in range(len(words) - gram + 1):
            word += [' '.join(words[i:i + gram])]
        return word
    return words

def writecsv(data,namadata,header):
    with open(namadata, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for key, value in data.items():
            writer.writerow([key,value])
            
def tweet2csv(data,namadata,header):
    with open(namadata, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for i in range(len(data.tweets)):
            writer.writerow([data.tweets[i],data.processed_tweets[i]])

def prosestext(text):
    text = sub_clean(text,pengganti)   
    stemfactory = StemmerFactory()
    stemmer = stemfactory.create_stemmer()
    text = stemmer.stem(text)
    #words = texts.split()
    return text

def praproses(text):
    text = sub_clean(text,pengganti) 
    text = wordfilter(text)
    return text

def data2csv(data1):
    writecsv(data1.tf_spam1,'/home/budi/tes/tfunigramspam.csv',
             ['1-gram Spam','Frekuensi'])
    writecsv(data1.tf_spam2,'/home/budi/tes/tfbigramspam.csv',
             ['Spam 2-gram Spam','Frekuensi'])
    writecsv(data1.tf_spam3,'/home/budi/tes/tftrigramspam.csv',
             ['3-gram Spam','Frekuensi'])
    writecsv(data1.tf_ham1,'/home/budi/tes/tfunigramham.csv',
             ['1-gram Ham','Frekuensi'])
    writecsv(data1.tf_ham2,'/home/budi/tes/tfbigramham.csv',
             ['2-gram Ham','Frekuensi'])
    writecsv(data1.tf_ham3,'/home/budi/tes/tftrigramham.csv',
             ['3-gram Ham','Frekuensi'])
    
    writecsv(data1.dfw1,'/home/budi/tes/dfunigram.csv',
             ['1-gram','Jumlah Tweet'])
    writecsv(data1.dfw2,'/home/budi/tes/dfbigram.csv',
             ['2-gram ','Jumlah Tweet'])
    writecsv(data1.dfw3,'/home/budi/tes/dftrigram.csv',
             ['3-gram','Jumlah Tweet'])
    writecsv(data1.pw_spam1,'/home/budi/tes/pwspam1.csv',
             ['1-gram','PW'])
    writecsv(data1.pw_ham1,'/home/budi/tes/pwham1.csv',
             ['1-gram','PW'])
   