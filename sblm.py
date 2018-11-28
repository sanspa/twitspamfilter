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

#%%
import sys

orig_stdout = sys.stdout
filename = 'data/output.txt'

f = open(filename,'w+')
sys.stdout = f
print('=====\n\n')
#%%
print('\n=====KFOLD====')
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)

for i, (train_index, test_index) in enumerate (kf.split(tweets,labels)):
    
    print('\n === FOLD KE ',i+1,' DARI ',kf.get_n_splits(),'===\n')
    if (min(test_index) > 0):
        if(max(test_index) < len(tweets)+1):
            print('Data Training index: 0 - ',min(test_index)-1,', ',
                  max(test_index)+1,' - ',max(train_index))
        else:
            print('Data Training index: 0 - ',max(train_index))
    else:
        print('Data Training index: ',min(train_index),' - ',max(train_index))
            
    print('Data Test index: ',min(test_index),' - ',max(test_index))
    clf = SpamClassifier(tweets[train_index],labels[train_index])
    clf.train()
    
    print('\n === 1 gram Bag-Of-Words ===')
    prediksi1 = clf.predict(tweets[test_index],'bow',1)
    clf.metrics(labels[test_index],prediksi1,tweets[test_index])
    
    print('\n === 2 gram ===')
    prediksi2 = clf.predict(tweets[test_index],'bow',2)
    clf.metrics(labels[test_index],prediksi2,tweets[test_index])
    print('\n === tfidf 1 gram===')
    prediksitfidf1 = clf.predict(tweets[test_index],'tfidf',1)
    clf.metrics(labels[test_index],prediksitfidf1,tweets[test_index])
    print('\n === tfidf  2 gram===')
    prediksitfidf2 = clf.predict(tweets[test_index],'tfidf',2)
    clf.metrics(labels[test_index],prediksitfidf2,tweets[test_index])
    print('\n === stupid backoff===')
    prediksistbo = clf.predict(tweets[test_index],'stbo',2)
    clf.metrics(labels[test_index],prediksistbo,tweets[test_index])
    
    
#%%
sys.stdout = orig_stdout
f.close()
print('===    \n')
    