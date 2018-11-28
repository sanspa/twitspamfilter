#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:36:45 2018

@author: budi
"""
from prosestext import *
from classifier import *
from substitusi import *
import pandas as pd

import numpy

corpus = pd.read_excel('tweets.xlsx')
corpus = corpus.sample(frac=1).reset_index(drop=True)
tweets = corpus['text'].values
labels = corpus['label'].values



train_index, test_index = list(), list()
for i in range(len(tweets)):
    if np.random.uniform(0,1) < 0.90:
        train_index += [i]
    else:
        test_index += [i]
        
clf = SpamClassifier(tweets[train_index],labels[train_index],method='bow',gram=1)
clf.train()    
prediksi = clf.predict(tweets[test_index])
metrics(labels[test_index],prediksi,tweets[test_index])
clf.printP()
#%%
corpus['text']= corpus['text'].str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
                                  ' urladdr ')

writer = pd.ExcelWriter('pd_clean.xlsx',engine='xlsxwriter')
corpus.to_excel(writer,sheet_name='Sheet1')
writer.save()
#%%
clf.classify('Ciptakan momen paling indah malam ini. LO HARUS MENANG @Persija_Jkt !! #PersijaDay')
#%%
clf.classify('RT @tatavir12: #expojogja sampai tgl 27 ya sayang ðŸ˜˜ðŸ˜˜ khusus hari ini ada spesial rate promo untuk 5slot aja #Trusted #expojogja #bbwjogja #â€¦')
#%%
from collections import Counter
print(clf.ham_tweets + clf.spam_tweets)
print(len(tweets[test_index]))
print(len(tweets[train_index]))
print(clf.ham_tweets)
print(clf.spam_tweets)
#%%
for value, count in Counter(clf.tf_ham).most_common(20):
    print(value,':',count)


print(clf.prob_spam_tweet)

#%%
for value, count in Counter(clf.tf_spam).most_common(20):
    print(value,':',count)


print(clf.prob_spam_tweet)
#%%
for value, count in Counter(clf.prob_ham).most_common(20):
    print(value,':',count)

print(clf.ham_words)
print(clf.vocab)
#%%
for value, count in Counter(clf.prob_spam).most_common(20):
    print(value,':',count)
    
print(clf.idf_ham['bangkit'])
print(clf.idf_spam['bangkit'])
print(clf.total_tweets)
print(clf.idf_spam.get('bangkit',0))

print(len(list(clf2.prob_ham.keys())))
print(len(clf.prob_ham))
#%%

clf2 = SpamClassifier(tweets[train_index],labels[train_index],method='tfidf',gram=1)
clf2.train()    

print(clf2.prob_ham['bangkit'])
print(len(clf2.prob_ham))
print(clf.total_tweets)
#%%

print(clf2.tf_spam['follow'])
print(clf2.idf_spam['follow'])
print(clf2.prob_spam['info'])
print(clf2.idf_ham['follow'])
print(clf2.idf_spam['follow'])
print(clf2.sum_tf_idf_ham)
print(len(list(clf2.prob_spam.keys())))

#%%
print(clf2.prob_ham['bangkit'])
print(clf2.prob_ham['menang'])
print(clf2.prob_ham['persija'])
print(clf2.prob_ham['ganjar'])
print(clf2.prob_ham['jkt'])
print(clf2.prob_ham['indonesia'])
print(clf2.prob_ham['ayo'])
print(clf2.prob_ham['nasional'])
print(clf2.prob_ham['selamat'])
print(clf2.prob_ham['bangsa'])
print(clf2.prob_ham['urladdr'])
print(clf2.prob_ham['main'])
print(clf2.prob_ham['tarawih'])
print(clf2.prob_ham['skor'])
print(clf2.prob_ham['persipura'])
print(clf2.prob_ham['can'])
print(clf2.prob_ham['persijaselamanya'])
print(clf2.prob_ham['semangat'])
print(clf2.prob_ham['pimpin'])
print(clf2.prob_ham['alhamdulillah'])


#%%
print(clf2.prob_spam['kak'])
print(clf2.prob_spam['info'])
print(clf2.prob_spam['follow'])
print(clf2.prob_spam['urladdr'])
print(clf2.prob_spam['bahasa'])
print(clf2.prob_spam['putar'])
print(clf2.prob_spam['ayo'])
print(clf2.prob_spam['folback'])
print(clf2.prob_spam['inggris'])
print(clf2.prob_spam['update'])

print(clf2.prob_spam['ajar'])
print(clf2.prob_spam['main'])
print(clf2.prob_spam['makassar'])
print(clf2.prob_spam['kursus'])
print(clf2.prob_spam['telp'])
print(clf2.prob_spam['pin7d07c8e6'])
print(clf2.prob_spam['keren'])
print(clf2.prob_spam['game'])
print(clf2.prob_spam['online'])
print(clf2.prob_spam['persibday'])

#%%
stemfactory = StemmerFactory()
stemmer = stemfactory.create_stemmer()
stop = StopWordRemoverFactory().create_stop_word_remover()
#%%
text = 'Ciptakan momen paling indah malam ini. LO HARUS MENANG @Persija_Jkt !! #PersijaDay'
clf.classify(text)
text = sub_clean(text,pengganti) 
print('cleaned',text)
text = stemmer.stem(text)
print('stemmed',text)
text = stop.remove(text)
print('stopword removed',text)
text = wordfilter(text)
print('filtering',text)
#%%
clf.classify('RT @tatavir12: #expojogja sampai tgl 27 ya sayang ðŸ˜˜ðŸ˜˜ khusus hari ini ada spesial rate promo untuk 5slot aja #Trusted #expojogja #bbwjogja #â€¦')
#%%
    
print(clf.prob_ham['cipta'])
print(clf.prob_ham['momen'])
print(clf.prob_ham['indah'])
print(clf.prob_ham['malam'])
print(clf.prob_ham['lo'])
print(clf.prob_ham['menang'])
print(clf.prob_ham['jkt'])

#%%
#print(clf.prob_spam['cipta'])
#print(clf.prob_spam['momen'])
#print(clf.prob_spam['indah'])
print(clf.prob_spam['malam'])
print(clf.prob_spam['lo'])
print(clf.prob_spam['menang'])
#print(clf.prob_spam['jkt'])
#%%
clfb = SpamClassifier(tweets[train_index],labels[train_index],method='bow',gram=2)
clfb.train() 

print(clfb.prob_ham)
#%%
text = 'Ciptakan momen paling indah malam ini. LO HARUS MENANG @Persija_Jkt !! #PersijaDay'
#print(clf.prob_ham['cipta momen'])
#print(clf.prob_ham['momen indah'])
#print(clf.prob_ham['indah malam'])
#print(clf.prob_ham['malam lo'])
#print(clf.prob_ham['lo menang'])
#print(clf.prob_ham['menang jkt'])

clfb.classify(text)
print(clfb.ham_words)
print(len(clfb.tf_ham))
print(len(list(clfb.prob_ham.keys())))

#%%
#print(clf.prob_spam['cipta momen'])
#print(clf.prob_spam['momen indah'])
#print(clf.prob_spam['indah malam'])
#print(clf.prob_spam['malam lo'])
#print(clf.prob_spam['lo menang'])
#print(clf.prob_spam['menang jkt'])

print(clfb.spam_words)
print(len(clfb.tf_spam))
print(len(list(clfb.prob_spam.keys())))
print(clfb.prob_ham_tweet)
print(clfb.prob_spam_tweet)
print(clf.prob_ham_tweet)
print(clf.prob_spam_tweet)
