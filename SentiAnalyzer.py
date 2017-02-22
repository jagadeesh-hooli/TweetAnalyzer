import nltk
from nltk.corpus import stopwords
import pandas as pd
import Features
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


#load data from CSV
trainDf = pd.read_csv("DATA/miniData50k.csv",quotechar="'",encoding='latin-1')
testDf = pd.read_csv("DATA/TestData.csv",quotechar="'",encoding='latin-1')
print("Data loaded from csv")


#get vocalbulary
vocabulary = Features.getVocabulary(trainDf["Tweet"])
print(len(vocabulary))
dict = {'neg':0,'pos':1}
#print(vocabulary)
print("vocabulay formed")


#Make trainingdata ready
def getTrainingData():
    trainingData = [(Features.tweet_tokenize(row[2]),dict[row[1]]) for row in trainDf.itertuples()]
    return trainingData

trainingData = getTrainingData()
print("Treaining data prepared")


#Prepare feature vector
cv = CountVectorizer(vocabulary=Features.getVocabulary(trainDf['Tweet']))
trainDataX = cv.fit_transform([tweet for tweet in trainDf['Tweet']])
testDataX = cv.fit_transform([tweet for tweet in testDf['Tweet']])

"""
transformer = TfidfVectorizer(vocabulary=Features.getVocabulary(trainDf['Tweet']))
trainDataX = transformer.fit_transform([tweet for tweet in trainDf['Tweet']]).todense().T
testDataX = transformer.fit_transform([tweet for tweet in testDf['Tweet']]).todense().T
"""

trainDataY = [dict[cls] for cls in trainDf['class']]
testDataY = [dict[cls] for cls in testDf['class']]

print("feature vector prepared")

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

#set the timer before running ML algo
startTiming = default_timer()

#Train the Data using classifier

classifier = getClassifier(1)
classifier.fit(trainDataX,trainDataY)
print("Classifier Trained")

#Get the predictive model
trainData_predictor = classifier.predict(trainDataX)
testData_predictor = classifier.predict(testDataX)

print("\n Time taken by ML algo for classification is: %.2f" % (default_timer() - startTiming))
print("\n ------------Sample tweet classification-----------------")
sample_tweets=["@PrincessSuperC Hey Cici sweetheart! Just wanted to let u know I luv u! OH! and will the mixtape drop soon? FANTASY RIDE MAY 5TH!!!!",
               "@Msdebramaye I heard about that contest! Congrats girl!!",
               "UNC!!! NCAA Champs!! Franklin St.: I WAS THERE!! WILD AND CRAZY!!!!!! Nothing like it...EVER http://tinyurl.com/49955t3",
               "Disappointing day. Attended a car boot sale to raise some funds for the sanctuary, made a total of 88p after the entry fee - sigh",
               "no more taking Irish car bombs with strange Australian women who can drink like rockstars...my head hurts.",
               "I am having terrible day!!"]

for tweet in sample_tweets:
    print(tweet+" => "+str(classifier.predict(cv.fit_transform([tweet]))).replace("[1]","Positive").replace("[0]","Negative"))
#print(classifier.predict(cv.fit_transform(["Today I am feeling so good"])))
#print(classifier.predict(cv.fit_transform(["This product is very bad"])))
#Evaluation metrics

"""On Training Data"""
print("\n ------------Training data Metrics----------")
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(trainDataY,trainData_predictor)))
print("\n---Confusion Matrix---")
print("{0}".format(metrics.confusion_matrix(trainDataY,trainData_predictor,labels=[1,0])))
print(" ")
print("Classification Report")
print(metrics.classification_report(trainDataY,trainData_predictor,labels=[1,0]))

"""ON Test Data"""
print("------------Test data Metrics----------")
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(testDataY,testData_predictor)))
print("---Confusion Matrix---")
print("{0}".format(metrics.confusion_matrix(testDataY,testData_predictor,labels=[1,0])))
print(" ")
print("Classification Report")
print(metrics.classification_report(testDataY,testData_predictor,labels=[1,0]))