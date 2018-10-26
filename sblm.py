#%%

from prosestext import *
from classifier import *
from substitusi import *
import pandas as pd

import numpy

corpus = pd.read_excel('tweets.xlsx')
corpus = corpus.sample(frac=1).reset_index(drop=True)
tweets = corpus['text'].values
labels = corpus['label'].values
#corpus.loc[:,'text'] = corpus['text'].apply(lambda x: sub_clean(x,pengganti))

#%%
#%%
print('\n=====KFOLD====')
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
print(kf)
for train_index, test_index in kf.split(tweets,labels):
    print('\n== 1. Unigram/Bag-of_Words ==')
    clf = SpamClassifier(tweets[train_index],labels[train_index],method='bow',gram=1)
    clf.train()    
    prediksi = clf.predict(tweets[test_index])
    metrics(labels[test_index],prediksi,tweets[test_index])
    
    print('\n== 2. Unigram/Bag-of_Words with tfidf ==')
    clf = SpamClassifier(tweets[train_index],labels[train_index],method='tfidf',gram=1)
    clf.train()    
    prediksi = clf.predict(tweets[test_index])
    metrics(labels[test_index],prediksi,tweets[test_index])
    
    print('\n== 3. Bigram ==')
    clf = SpamClassifier(tweets[train_index],labels[train_index],method='bow',gram=2)
    clf.train()    
    prediksi = clf.predict(tweets[test_index])
    metrics(labels[test_index],prediksi,tweets[test_index])
    
    print('\n== 4. Bigram with tfidf ==')
    clf = SpamClassifier(tweets[train_index],labels[train_index],method='tfidf',gram=2)
    clf.train()    
    prediksi = clf.predict(tweets[test_index])
    metrics(labels[test_index],prediksi,tweets[test_index])
    
    print('\n== 5. Stupid Backoff ==')
    clf = SpamClassifier(tweets[train_index],labels[train_index],method='stbo',gram=2)
    clf.train()    
    prediksi = clf.predict(tweets[test_index])
    metrics(labels[test_index],prediksi,tweets[test_index])




#%%
