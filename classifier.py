#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:56:00 2018

@author: budi
"""

import matplotlib.pyplot as plt
from math import log10
import random
from  prosestext import *
import pandas as pd
import datetime
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
from substitusi import *

class SpamClassifier(object):
    def __init__(self, tweets,labels):
        self.tweets, self.labels = tweets,labels
        self.clean_tweets = []
        self.conv_tweets = []
        self.stem_tweets = []
        self.processed_tweets = []

        self.spam_tweets, self.ham_tweets = (labels==1).sum(),(labels==0).sum()
        self.total_tweets = len(self.tweets)
        
        self.testdata = []
        self.testdata_terproses = []
        
        self.vocab1 = list()
        self.vocab2 = list()
        self.vocab3 = list()
        
             
        self.prior_spam = 0.0
        self.prior_ham = 0.0
        
        self.tf_spam1 = dict()
        self.tf_ham1 = dict()
        self.tf_spam2 = dict()
        self.tf_ham2 = dict()
        self.tf_spam3 = dict()
        self.tf_ham3 = dict()
        
        self.dfw1 = dict()
        self.dfw2 = dict()
        self.dfw3 = dict()
        
        self.pwtfidf_spam1 = dict()
        self.pwtfidf_ham1 = dict()
        self.pwtfidf_spam2 = dict()
        self.pwtfidf_ham2 = dict()
        self.pwtfidf_spam3 = dict()
        self.pwtfidf_ham3 = dict()
        
        
        self.stemmer = StemmerFactory().create_stemmer()
        self.stop = StopWordRemoverFactory().create_stop_word_remover()
        self.stop.dictionary.add('lasturladdr')
        self.stop.dictionary.add('rt')
    
    
    def praproses(self):
        for i in range(len(self.tweets)):
            self.clean_tweets.append(clean_text(self.tweets[i]))
            self.conv_tweets.append(konversi(self.clean_tweets[i],pengganti))
            self.stem_tweets.append(self.stemmer.stem(self.conv_tweets[i]))
            self.processed_tweets.append(self.stop.remove(self.stem_tweets[i]))
        
               
    def praprosestext(self,teks):
        cteks = clean_text(teks)
        konv_teks = konversi(cteks,pengganti)
        stemteks = self.stemmer.stem(konv_teks)
        nosw_teks = self.stop.remove(stemteks)
        return nosw_teks
    
    
    def hitungTFDF(self):
        for i in range(self.total_tweets):
            
            dfw = {}
            tfunigram = createToken(self.processed_tweets[i],gram=1)
            for word in tfunigram:
                
                if dfw.get(word,0)== 0:  # hitung dokumen berisi word
                    dfw[word] = 1
                    self.dfw1[word] = self.dfw1.get(word,0) + 1  
                
                if self.labels[i]:
                    self.tf_spam1[word] = self.tf_spam1.get(word,0) + 1
                                      
                else:
                    self.tf_ham1[word] = self.tf_ham1.get(word,0) + 1                  
                    
            dfw = {}
            tfbigram = createToken(self.processed_tweets[i],gram=2)
            for word in tfbigram:
                
                if dfw.get(word,0)== 0:  # hitung dokumen berisi word
                    dfw[word] = 1
                    self.dfw2[word] = self.dfw2.get(word,0) + 1
                    
                if self.labels[i]:
                    self.tf_spam2[word] = self.tf_spam2.get(word,0) + 1
                    
                else:
                    self.tf_ham2[word] = self.tf_ham2.get(word,0) + 1               
                    

            dfw = {}
            tftrigram = createToken(self.processed_tweets[i],gram=3)
            for word in tftrigram:
                
                if dfw.get(word,0)== 0:  # hitung dokumen berisi word
                    dfw[word] = 1
                    self.dfw3[word] = self.dfw3.get(word,0) + 1
                
                if self.labels[i]:
                    self.tf_spam3[word] = self.tf_spam3.get(word,0) + 1
                    
                else:
                    self.tf_ham3[word] = self.tf_ham3.get(word,0) + 1
                        
        self.vocab1 = list(dict(self.tf_spam1,**self.tf_ham1).keys())
        self.vocab2 = list(dict(self.tf_spam2,**self.tf_ham2).keys())
        self.vocab3 = list(dict(self.tf_spam3,**self.tf_ham3).keys())
                    

    
    def train(self):
        self.praproses()
        self.hitungTFDF()
        
        self.prior_spam = self.spam_tweets/ self.total_tweets
        self.prior_ham = self.ham_tweets / self.total_tweets       
        
        #****Hitung tfidf untuk 1-gram dan 2-gram***
        
         #***** 1-gram****
        for word in self.tf_spam1:
            self.pwtfidf_spam1[word] = self.tf_spam1[word] \
                   * log10(len(self.tweets) / self.dfw1[word])
                    
        for word in self.tf_ham1:
            self.pwtfidf_ham1[word] = self.tf_ham1[word] \
                   * log10(len(self.tweets) / self.dfw1[word])
                               
         #===2 gram===
        for word in self.tf_spam2:
            self.pwtfidf_spam2[word] = self.tf_spam2[word] \
                    * log10(len(self.tweets) / (self.dfw2[word]))            
                    
        for word in self.tf_ham2:
            self.pwtfidf_ham2[word] = self.tf_ham2[word] \
                    * log10(len(self.tweets) / self.dfw2[word])
               
        
    def classify1(self,text,metode):
        
        self.metode = metode+'1gr'
        proses_text = self.praprosestext(text)   
        self.testdata_terproses.append(proses_text)
        token = createToken(proses_text,gram=1)
        
        pSpam = log10(self.prior_spam)
        pHam = log10(self.prior_ham)
        
        for word in token: 
            
            #==hitung probbilitas spam                  
            if metode == 'tfidf':
                pSpam += log10(self.pwtfidf_spam1.get(word,1) + 1)
                pSpam -= log10(sum(self.pwtfidf_spam1.values()) 
                             + len(self.tf_spam1))
            if metode == 'bow':
                pSpam += log10((self.tf_spam1.get(word,0) + 1) 
                            /(sum(self.tf_spam1.values()) + len(self.vocab1)))
                                
           #== Hitung untuk ham =====  
            if metode == 'tfidf':
                pHam += log10(self.pwtfidf_ham1.get(word,1) + 1)
                pHam -= log10(sum(self.pwtfidf_ham1.values()) 
                             + len(self.tf_ham1))
            if metode == 'bow':
                pHam += log10((self.tf_ham1.get(word,0) + 1)
                            /(sum(self.tf_spam1.values()) + len(self.vocab1)))   
              
                
        #print("pSpam: ",pSpam," pHam: ",pHam)        
        return pSpam >= pHam
    
    def classify2(self, text,metode):
        
        self.metode = metode+'2gr'
        proses_text = self.praprosestext(text) 
        self.testdata_terproses.append(proses_text)
        token = createToken(proses_text,gram=2)
        
        pSpam = log10(self.prior_spam)
        pHam = log10(self.prior_ham)
        
        for word in token: 
            
            #==hitung probbilitas spam                  
            if metode == 'tfidf':
                pSpam += log10(self.pwtfidf_spam2.get(word,1) + 1)
                pSpam -= log10(sum(self.pwtfidf_spam2.values()) 
                             + len(self.tf_spam2))
            else:
                pSpam += log10((self.tf_spam2.get(word,0) + 1) 
                             /(sum(self.tf_spam2.values()) + len(self.vocab2)))
                
           #== Hitung untuk ham =====  
            if metode == 'tfidf':
                pHam += log10(self.pwtfidf_ham2.get(word,1) + 1)
                pHam -= log10(sum(self.pwtfidf_ham2.values()) 
                             + len(self.tf_ham2))
            else:
                pHam += log10((self.tf_ham2.get(word,0) + 1)
                             /(sum(self.tf_spam2.values()) + len(self.vocab2)))
        
        #print('pSpam: ',pSpam,' pHam: ',pHam)
        return pSpam >= pHam

    
    def sbclassify(self,text):
        
        self.metode = 'stupidbackoff'
        proses_text = self.praprosestext(text) 
        self.testdata_terproses.append(proses_text)
        hamscore = 0.0
        spamscore = 0.0
        
        words = createToken(proses_text,gram=2)
        
        for word in words:       
            wordtoken = word.split()
            tokenprev = wordtoken[0]
            tokennext = wordtoken[1]
            if word in self.tf_ham2:
                bicount = self.tf_ham2[word]
                bi_unicount = self.tf_ham1[tokenprev]
                hamscore += log10(bicount)
                hamscore -= log10(bi_unicount)
            else:
                if tokennext in self.tf_ham1:
                    unicount = self.tf_ham1[tokennext]
                else:
                    unicount = 0.4
                hamscore += log10(0.4)
                hamscore += log10(unicount)
                hamscore -= log10(sum(self.tf_ham1.values())+ len(self.vocab1))
        
            if word in self.tf_spam2:
                bicount2 = self.tf_spam2[word]
                bi_unicount2 = self.tf_spam1[tokenprev]
                spamscore += log10(bicount2)
                spamscore -= log10(bi_unicount2)
            else:
                if tokennext in self.tf_spam1:
                    unicount2 = self.tf_spam1[tokennext]
                else:
                    unicount2 = 0.4
                spamscore += log10(0.4)
                spamscore += log10(unicount2)
                spamscore -= log10(sum(self.tf_spam1.values()) + len(self.vocab1))     
            #spamscore += log10(self.prior_spam)
            #hamscore += log10(self.prior_ham)
        return spamscore >= hamscore
        
    
    def predict(self, test_data,metode,gram):
        '''metode = stbo = stupid backoff'
                    bow = bag off word
                    tfidf = with tfidf
        '''
        self.testdata = []
        self.testdata_terproses = []
        result = dict()
        if metode == 'stbo':
            for (i, tweet) in enumerate(test_data):
                result[i] = int(self.sbclassify(tweet))
        else:       
            if gram == 1:
                for (i, tweet) in enumerate(test_data):
                    result[i] = int(self.classify1(tweet,metode))
            if gram == 2:
                for (i, tweet) in enumerate(test_data):
                    result[i] = int(self.classify2(tweet,metode))                
        return result
      

    def metrics(self, labels, predictions,tweets):
        etext=[]
        eptext=[]
        elabel=[]
        true_pos, true_neg, false_pos, false_neg = 0,0,0,0
        for i in range(len(labels)):
            true_pos += int((labels[i] == 1) and (predictions[i] == 1))
            true_neg += int(labels[i] == 0 and predictions[i] == 0)
            if (labels[i] == 0 and predictions[i] == 1):
                false_pos += 1
                etext.append(tweets[i])
                eptext.append(self.praprosestext(tweets[i]))
                elabel.append('fp')
            if (labels[i] == 1 and predictions[i] == 0):
                false_neg += 1
                etext.append(tweets[i])
                eptext.append(self.praprosestext(tweets[i]))
                elabel.append('fn')
        edf = pd.DataFrame(list(zip(etext,eptext,elabel)),columns=['text','stemmedtext','label'])
        filename = 'data/false_'+self.metode+ '.xlsx'
        writer = pd.ExcelWriter(filename,engine='xlsxwriter')
        edf.to_excel(writer,sheet_name='Sheet1')
        writer.save()
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        Fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy) 
        print('\n==Confusion Matrix===')
        print("True Positiv: ",true_pos)
        print("False Positiv: ",false_pos)
        print("True Negativ: ",true_neg)
        print("False Negativ: ",false_neg)
    
    
