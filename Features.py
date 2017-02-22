import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
from nltk.probability import FreqDist
from heapq import nlargest
from sklearn.feature_extraction.text import CountVectorizer



#Extract sentimental score and Tweet into file
'''
with open("DATA/pro1.csv",'w') as f1:
    for row in file:
        temp = row.split(',',5)
        print >>f1, temp[0].replace('"','')+","+str(temp[5]).replace('"',"'")


'''


#lib:gets tokenizes tweet sentences to words
def tweet_tokenize(tweet):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(tweet)

#lib:gets stopwords in English(e.g: 'is,was etc')
def getStopWords():
    return set(stopwords.words('english'))


#lib:returns regular expression of passed pattern Type
def getRegualarExpression(patternType):
    regExpn ={
        'hashWords': '(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)',
        'userNames': '(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9]+)',
        'urls': 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    }
    return regExpn.get(patternType)

#move fn to lib
def getCleanedData(text):
   customStopwords = getStopWords()
   regexes = {
       re.compile(getRegualarExpression('hashWords')),
       re.compile(getRegualarExpression('userNames')),
       re.compile(getRegualarExpression('urls')),
       }
   cleanedText = [word for word in text
                  if word not in customStopwords and
                  word.isalpha() and
                  not any(regex.match(word) for regex in regexes)]
   return cleanedText


#Formulate vocalbulary:Returns top 20% frequent words
def getVocabulary(tweetDataFrame):
    vocabulary = []
    for tweet in tweetDataFrame:
        for word in tweet_tokenize(tweet):
            vocabulary.append(word)
    cleanedVocab = getCleanedData(vocabulary)
    freq = FreqDist(cleanedVocab)
    return nlargest(int(round(((len(cleanedVocab)*20.0)/100))),freq,key=freq.get)


#Formulate vocalbulary:Returns top 20% frequent words
def getVocabulary(tweetDataFrame):
    vocabulary = []
    for tweet in tweetDataFrame:
        for word in tweet_tokenize(tweet):
            vocabulary.append(word)
    cleanedVocab = getCleanedData(vocabulary)
    freq = FreqDist(cleanedVocab)
    return nlargest(int(round(((len(cleanedVocab)*20.0)/100))),freq,key=freq.get)



"""Below code is only used for identifying important words in a given data set"""
#load data
"""
def getLoadedDataFrame():
    return pd.read_csv("DATA/miniData50k.csv",quotechar="'",encoding='latin-1')

df= getLoadedDataFrame()

cv = CountVectorizer(vocabulary=getVocabulary(df['Tweet']))
X = cv.fit_transform([tweet for tweet in df['Tweet']])
Y = [dict[cls] for cls in df['class']]


dict = {'neg':0,'pos':1}

model = ExtraTreesClassifier()
model.fit(X, Y)
feature_words= cv.get_feature_names()
feature_score = model.feature_importances_
wordScorelist =  zip(feature_words,feature_score)
#print(model.n_features_)

"""



