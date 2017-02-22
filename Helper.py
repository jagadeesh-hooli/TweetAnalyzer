from nltk.tokenize import TweetTokenizer
import nltk

def tweet_tokenize(tweet):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(tweet)

def extract_pattern(tweet,pattern):
    word_text = tweet_tokenize(tweet)
    tagged = nltk.pos_tag(word_text)
    NPChunker = nltk.RegexpParser(pattern)
    result = NPChunker.parse(tagged)
    phrases = []
    p = ''
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        A = subtree.leaves()
        p = ''
        for  item in A:
            p += item[0] + " "
        phrases.append(p)

    return phrases