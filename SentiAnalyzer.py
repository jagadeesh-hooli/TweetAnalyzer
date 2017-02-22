import nltk
from nltk.corpus import stopwords
import pandas as pd
from Features import getVocabulary,tweet_tokenize
import sklearn
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer

from sklearn.ensemble import ExtraTreesClassifier
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
from nltk.probability import FreqDist
from heapq import nlargest

"""
#load data from CSV
with open("DATA/training.1600000.processed.noemoticon.csv",'r') as f:
    file = f.readlines()



# List with two cols: sentimental score and Tweet
testdata = []
for row in file:
    temp = row.split(',',5)
    testdata.append([temp[0],str(temp[5]).replace('"','')])


#Formulate vocalbulary
def getVocabulary():
    vocabulary = list(set([word for line in testdata for word in line[1].split()]))
    return vocabulary

vocabulary = getVocabulary()
print("WordLen before remiving stopwords:%d" %len(vocabulary))

#remove stopwords
customStopwords = set(stopwords.words('english'))
vocabulary = [word for word in vocabulary if word not in customStopwords]
print("WordLen after remiving stopwords:%d" %len(vocabulary))


#Make trainingdata ready
def getTrainingData():
    trainingData = [(str(line[1]).split(),str(line[0]).replace('"','')) for line in testdata]
    return trainingData

trainingData = getTrainingData()
#print(trainingData)


#Prepare feature vector
def extract_features(review):
  review_words=set(review)
  features={}
  for word in vocabulary:
      features[word]=(word in review_words)
  return features

print('feature vector prepared')

#Get the classifier
def getTrainedNBClassifier(extract_features,trainingData):
    print('inside nbctran')
    trainingFeatures = nltk.classify.apply_features(extract_features,trainingData)
    print('feat extr')

    trainedNBClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

    print('classfy')
    return trainedNBClassifier


trainedNBClassifier = getTrainedNBClassifier(extract_features, trainingData)
print('classifier trained')

#Perform sentimental analysis
def naiveBayesSentimentCalculator(review):
    problemInstance = review.split()
    problemFeatures = extract_features(problemInstance)
    return trainedNBClassifier.classify(problemFeatures)
print ('end of nb fn')

print(naiveBayesSentimentCalculator("fear of failure"))
print("End of prog")
"""



#Gets test data from CSV
def getLoadedTestDataFrame():
    return pd.read_csv("DATA/TestData.csv", quotechar="'", encoding='latin-1')


#Gets train data from CSV
def getLoadedTrainingDataFrame():
    return pd.read_csv("DATA/miniData50k.csv",quotechar="'",encoding='latin-1')


#returns mapping value of neg, pos labels
def getsClassValueFromLabel(classType):
    dict = {'neg': 0, 'pos': 1}
    return dict.get(classType)


#Make trainingdata ready
def getTrainingData():
    trainingData = [(tweet_tokenize(row[2]),getsClassValueFromLabel(row[1])) for row in trainDf.itertuples()]
    return trainingData


# returns the countVectorizer funtion
def getCountVectorizer(vocabulary):
    return CountVectorizer(vocabulary=vocabulary)


#Prepare feature vector
def getFeatureVector(featureVectorizer,tweetDataFrame,classDataFrame):
    xVector = featureVectorizer.fit_transform([tweet for tweet in tweetDataFrame])
    yVector = [getsClassValueFromLabel(cls) for cls in classDataFrame]
    return xVector,yVector


#prepare list of supervised machine learning classifiers
def getClassifier(classifierCode):
    classifiers ={
        1:BernoulliNB(),
        2:ExtraTreesClassifier(),
        3:DecisionTreeClassifier(),
        4:LogisticRegression(),
        5:GaussianNB(),
        6:SVC(),
        7:RandomForestClassifier(),

    }
    return classifiers.get(classifierCode)


#get trained classifier
def getTrainedClassifier(dataX,dataY):
    classifier = getClassifier(1)
    return classifier.fit(trainDataX, trainDataY)


#logging Evaluation metrics
def logDiagnostics(dataY,dataPredictor,diagonisticsType):
    if(diagonisticsType==1):
        print("\n ------------Training data Metrics----------")
    else:
        print("\n------------Test data Metrics----------")
    print("Accuracy:{0:.4f}".format(metrics.accuracy_score(dataY, dataPredictor)))
    print("\n---Confusion Matrix---")
    print("{0}".format(metrics.confusion_matrix(dataY, dataPredictor, labels=[1, 0])))
    print(" ")
    print("Classification Report")
    print(metrics.classification_report(dataY, dataPredictor, labels=[1, 0]))


#test sentimental prediction for sample tweets
def doPredictiveTestforSampleTweets(trainedClassifier,featureVectorizer):
    print("\n ------------Sample tweet classification-----------------")
    sample_tweets=["@PrincessSuperC Hey Cici sweetheart! Just wanted to let u know I luv u! OH! and will the mixtape drop soon? FANTASY RIDE MAY 5TH!!!!",
               "@Msdebramaye I heard about that contest! Congrats girl!!",
               "UNC!!! NCAA Champs!! Franklin St.: I WAS THERE!! WILD AND CRAZY!!!!!! Nothing like it...EVER http://tinyurl.com/49955t3",
               "Disappointing day. Attended a car boot sale to raise some funds for the sanctuary, made a total of 88p after the entry fee - sigh",
               "no more taking Irish car bombs with strange Australian women who can drink like rockstars...my head hurts.",
               "I am having terrible day!!"]

    for tweet in sample_tweets:
        print(tweet+" => "+str(trainedClassifier.predict(featureVectorizer.fit_transform([tweet]))).replace("[1]","Positive").replace("[0]","Negative"))


testDf = getLoadedTestDataFrame()
trainDf = getLoadedTrainingDataFrame()
print("--Data loaded from csv--")


vocabulary = getVocabulary(trainDf["Tweet"])
#print(vocabulary)
print("--Vocabulay formed and has %d words--" %len(vocabulary))

trainingData = getTrainingData()
print("--Training data prepared--")

"""transformer = TfidfVectorizer(vocabulary=Features.getVocabulary(trainDf['Tweet']))"""
cv = getCountVectorizer(getVocabulary(trainDf['Tweet']))
trainDataX,trainDataY = getFeatureVector(cv,trainDf['Tweet'],trainDf['class'])
testDataX,testDataY = getFeatureVector(cv,testDf['Tweet'],testDf['class'])
print("feature vector prepared")

#set the timer before training ML algo
startTiming = default_timer()
trainedClassifier = getTrainedClassifier(trainDataX,trainDataY)
print("Classifier Trained")

#Get the predictive model
trainData_predictor = trainedClassifier.predict(trainDataX)
testData_predictor = trainedClassifier.predict(testDataX)
print("\n Time taken by ML algo for classification is: %.2f" % (default_timer() - startTiming))

doPredictiveTestforSampleTweets(trainedClassifier,cv)
logDiagnostics(trainDataY,trainData_predictor,1)
logDiagnostics(testDataY,testData_predictor,0)




