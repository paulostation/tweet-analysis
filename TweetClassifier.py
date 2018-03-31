import pandas as pd
import numpy as np
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

from nltk.corpus import stopwords
stopset = set(stopwords.words('portuguese'))

# global variable used as classifier
classifier = None

# Filter non-ascii characters because Bayes classifier cannot show non-ascii on show_most_informative_features()
def filter_non_ascii(word):
    return re.sub(r'[^\x00-\x7f]', r'', word)


def word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])

# Function used to classify tweets
def classify(text):
    features = word_feats(text.split())
    classification = classifier.classify(features)
    if classification == "pos":
        return "positive"
    else:
        return "negative"

df = pd.read_csv('datasets/labeled_tweets.csv', names=['sentiment', 'text']);

# separate positive tweets only
positive_tweets = df.loc[df['sentiment'] == 'positive']['text'].tolist()
# separate negative tweets only
negative_tweets = df.loc[df['sentiment'] == 'negative']['text'].tolist()

positive_tweets = map(filter_non_ascii, positive_tweets)
negative_tweets = map(filter_non_ascii, negative_tweets)


posfeats = [(word_feats(tweet.split()), 'pos') for tweet in positive_tweets]
negfeats = [(word_feats(tweet.split()), 'neg') for tweet in negative_tweets]
# separate 75% for train, 25% for test
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (
    len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()