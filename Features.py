import nltk
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
import Helper
from nltk.tokenize import TweetTokenizer
import codecs
from nltk.corpus import stopwords
import re
from nltk.probability import FreqDist
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


#load data
df = pd.read_csv("DATA/miniData50k.csv",quotechar="'",encoding='latin-1')



#Extract sentimental score and Tweet into file
'''
with open("DATA/pro1.csv",'w') as f1:
    for row in file:
        temp = row.split(',',5)
        print >>f1, temp[0].replace('"','')+","+str(temp[5]).replace('"',"'")


'''






#move fn to lib
def tweet_tokenize(tweet):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(tweet)


#move fn to lib
def getCleanedData(text):
   customStopwords = set(stopwords.words('english'))
   regexes = [re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)'),
              re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9]+)'),
              re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
              ]
   cleanedText = [word for word in text
                  if word not in customStopwords and
                  word.isalpha() and
                  not any(regex.match(word) for regex in regexes)]
   return cleanedText


#Formulate vocalbulary
def getVocabulary(tweetDataFrame):
    vocabulary = []
    for tweet in tweetDataFrame:
        for word in tweet_tokenize(tweet):
            vocabulary.append(word)
    cleanedVocab = getCleanedData(vocabulary)
    freq = FreqDist(cleanedVocab)
    return nlargest(int(round(((len(cleanedVocab)*20.0)/100))),freq,key=freq.get)






cv = CountVectorizer(vocabulary=getVocabulary(df['Tweet']))
X = cv.fit_transform([tweet for tweet in df['Tweet']])

feature_words= cv.get_feature_names()
dict = {'neg':0,'pos':1}
Y = [dict[cls] for cls in df['class']]


model = ExtraTreesClassifier()
model.fit(X, Y)
feature_score = model.feature_importances_
wordScorelist =  zip(feature_words,feature_score)
model.n_features_





