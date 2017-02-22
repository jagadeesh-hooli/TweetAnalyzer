from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re


class Library(object):

    def __init__(self, features, feature_score_type="term_frequency"):
        self.features = features
        self.feature_score_type = feature_score_type


    def tweet_tokenize(tweet):
        tknzr = TweetTokenizer()
        return tknzr.tokenize(tweet)


    def getCleanedData(text):
        customStopwords = set(stopwords.words('english'))
        cleanedText = [word for word in text if word not in customStopwords]
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
        cleanedText = list(set(word for word in cleanedText if word not in urls))
        return cleanedText


    def extract_features(self, text):
        """ Given a tweet, this function extracts the value of predefined features """
        tokens = self.tweet_tokenize(text)
        feature_set = {}

        if self.feature_score_type == "presence":
            for feature in self.features:
                if feature in tokens:
                    feature_set[feature] = 1
                else:
                    feature_set[feature] = 0

        elif self.feature_score_type == "term_frequency":
            for feature in self.features:
                feature_set[feature] = tokens.count(feature)

        return feature_set
