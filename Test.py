from sklearn.externals import joblib
import csv


#test sentimental prediction for sample tweets
def doPredictiveTestforSampleTweets(trainedClassifier,featureVectorizer):
    print("\n ------------Sample tweet classification-----------------")
    sample_tweets=["@PrincessSuperC Hey Cici sweetheart! Just wanted to let u know I luv u! OH! and will the mixtape drop soon? FANTASY RIDE MAY 5TH!!!!",\
               "@Msdebramaye I heard about that contest! Congrats girl!!",
               "UNC!!! NCAA Champs!! Franklin St.: I WAS THERE!! WILD AND CRAZY!!!!!! Nothing like it...EVER http://tinyurl.com/49955t3",
               "Disappointing day. Attended a car boot sale to raise some funds for the sanctuary, made a total of 88p after the entry fee - sigh",
               "no more taking Irish car bombs with strange Australian women who can drink like rockstars...my head hurts.",
               "I am having terrible day!!","Excited for tomarrow.. Yeah!"]

    for tweet in sample_tweets:
        print(tweet+" => "+str(trainedClassifier.predict(featureVectorizer.fit_transform([tweet]))).replace("[1]","Positive").replace("[0]","Negative"))


#Read the tweets and do the prediction
def predictSentiScoreForTweets(trainedClassifier,featureVectorizer):
    # load raw tweets from CSV
    with open("/home/jagadeesh/PycharmProjects/TweetAnalyzer/TempDatafiles/RawTweetsTemp.csv", 'r') as f:
        file = f.readlines()

    with open("/home/jagadeesh/PycharmProjects/TweetAnalyzer/TempDatafiles/ClassifiedTweetsTemp.csv", 'w') as f1:
        for row in file:
            print >> f1,str(trainedClassifier.predict(featureVectorizer.fit_transform([row]))).replace("[1]","1").replace("[0]","0")+","+row ,


trained_model =joblib.load("/home/jagadeesh/PycharmProjects/TweetAnalyzer/TempDatafiles/trained_model.sav")
featureVectorizer = joblib.load("/home/jagadeesh/PycharmProjects/TweetAnalyzer/TempDatafiles/featureVectorizer.sav")

#doPredictiveTestforSampleTweets(trained_model,featureVectorizer)
#predictSentiScoreForTweets(trained_model,featureVectorizer)

def csv_dict_reader(file_obj,trainedClassifier,featureVectorizer):
    """
    Read a CSV file using csv.DictReader
    """
    reader = csv.DictReader(file_obj, delimiter=',')
    outfilePath="/home/jagadeesh/PycharmProjects/TweetAnalyzer/TempDatafiles/ClassifiedTweetsTemp.csv"
    with open(outfilePath, "w") as csv_file:
        csv_file.seek(0)
        csv_file.truncate()
        print >> csv_file ,"Id"+","+"Topic"+","+"DateOfTweet"+","+"SentiScore"+","+"Desc"
        for line in reader:
            print >> csv_file, line["Id"]+","+line["Topic"]+","+line["DateOfTweet"]+","+str(trainedClassifier.predict(featureVectorizer.fit_transform([line["Desc"]]))).replace("[1]","1").replace("[0]","0")+","+line["Desc"]

        # print(line["Id"]),
        # print(line["Topic"]),
        # print(line["DateOfTweet"]),
        # print(line["Desc"])




if __name__ == "__main__":
    with open("/home/jagadeesh/PycharmProjects/TweetAnalyzer/TempDatafiles/RawTweetsTemp.csv") as f_obj:
        csv_dict_reader(f_obj,trained_model,featureVectorizer)


