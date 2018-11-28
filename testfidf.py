#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 04:40:17 2018

@author: budi
"""

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
        
clf = SpamClassifier(tweets[train_index],labels[train_index],method='tfidf',gram=1)
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
print(clf.sum_tf_idf_ham)
print(len(list(clf.prob_spam.keys())))
print(len(clf.prob_ham))
