from sklearn.externals import joblib


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


classifierFileName = 'trained_model.sav'
featureVectorizerFileName = 'featureVectorizer.sav'
trained_model =joblib.load(classifierFileName)
featureVectorizer = joblib.load(featureVectorizerFileName)

doPredictiveTestforSampleTweets(trained_model,featureVectorizer)
