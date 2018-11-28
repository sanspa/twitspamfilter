#%%

import prosestext
from classifier import *
from substitusi import *
from evaluate import *
import pandas as pd


import numpy

corpus = pd.read_excel('tweets.xlsx')
corpus = corpus.sample(frac=1).reset_index(drop=True)
tweets = corpus['text'].values
labels = corpus['label'].values
#corpus.loc[:,'text'] = corpus['text'].apply(lambda x: sub_clean(x,pengganti))

#%%

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(tweets, labels, 
                                                    train_size=0.9,
                                                    test_size=0.1,
                                                    random_state=123)
#%%
#import sys
#orig_stdout = sys.stdout
#
#f = open('/home/budi/tes/out.txt','w+')
#sys.stdout = f

#%%
clf = SpamClassifier(train_X,train_y)
clf.train()
#%%
print("====Original Tweets=====")
print(clf.tweets[:5])
print("\n======1. Clean html escape, spasi, dan ganti baris======")
print(clf.clean_tweets[:5])
print("\n======2. KOnversi\substitusi kata======")
print(clf.conv_tweets[:5])

print("\n======3. Stemming ======")
print(clf.stem_tweets[:5])
print("\n======4. Remove Stop Word ======")
print(clf.processed_tweets[:5])

#%%  
print('\n === 1 gram Bag-Of-Words ===')
prediksi1 = clf.predict(test_X,'bow',1)
clf.metrics(test_y,prediksi1,test_X)
print('\n === 2 gram ===')
prediksi2 = clf.predict(test_X,'bow',2)
clf.metrics(test_y,prediksi2,test_X)
print('\n === tfidf 1 gram===')
prediksitfidf1 = clf.predict(test_X,'tfidf',1)
clf.metrics(test_y,prediksitfidf1,test_X)
print('\n === tfidf  2 gram===')
prediksitfidf2 = clf.predict(test_X,'tfidf',2)
clf.metrics(test_y,prediksitfidf2,test_X)
print('\n === stupid backoff===')
prediksistbo = clf.predict(test_X,'stbo',2)
clf.metrics(test_y,prediksistbo,test_X)

#%%
'''
print('\n=====KFOLD====')
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
print(kf)
for train_index, test_index in kf.split(tweets,labels):
    
    print('\n== 3. Bigram ==')
    clf = SpamClassifier(tweets[train_index],labels[train_index],method='bow',gram=2)
    clf.train()    
    prediksi = clf.predict(tweets[test_index])
    clf.metrics(labels[test_index],prediksi,tweets[test_index])
    
    print('\n== 4. Bigram with tfidf ==')
    clf = SpamClassifier(tweets[train_index],labels[train_index],method='tfidf',gram=2)
    clf.train()    
    prediksi = clf.predict(tweets[test_index])
    clf.metrics(labels[test_index],prediksi,tweets[test_index])
    #print(clf.tf_spam[0])

'''



##%%
#from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
#from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#stemfactory = StemmerFactory()
#stemmer = stemfactory.create_stemmer()
#stop = StopWordRemoverFactory().create_stop_word_remover()
#print(tweets[0])

#%%


#%%
#sys.stdout = orig_stdout
#f.close()