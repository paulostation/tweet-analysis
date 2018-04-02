import pandas as pd
import numpy as np
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

# global variable used as classifier
classifier = None

stopwords = set(["sobre","se","voc","isso","q","do","vai","pra","a","o","dilma","lula","bolsonaro","no","quando","cara","uma","so","um","da","ato","faz","as","esto","mais","com","foi","mesmo","das","quer","fazer","eu"])

# Filter non-ascii characters because Bayes classifier cannot show non-ascii on show_most_informative_features()
def filter_non_ascii(word):
    return re.sub(r'[^\x00-\x7f]', r'', word)

def filter_symbols(word):
    return re.sub(r'[^A-z\ ]|(https:\/\/[^\s]+)', r'', word)

def word_feats(words):
    return dict([(word, True) for word in words if word.lower() not in stopwords])

# Function used to classify tweets
def classify(text):
    features = word_feats(text.split())
    classification = classifier.classify(features)
    return classification

df = pd.read_csv('datasets/labeled_tweets.csv', names=['text', 'sentiment']);

# separate positive tweets only
positive_tweets = df.loc[df['sentiment'] == 'positive']['text'].tolist()
# separate negative tweets only
negative_tweets = df.loc[df['sentiment'] == 'negative']['text'].tolist()
# separate neutral tweets only
neutral_tweets = df.loc[df['sentiment'] == 'neutral']['text'].tolist()

positive_tweets = map(filter_non_ascii, positive_tweets)
negative_tweets = map(filter_non_ascii, negative_tweets)
neutral_tweets = map(filter_non_ascii, negative_tweets)

positive_tweets = positive_tweets[:69]
negative_tweets = negative_tweets[:69]
neutral_tweets = neutral_tweets[:69]

posfeats = [(word_feats(tweet.split()), 'positive') for tweet in positive_tweets]
negfeats = [(word_feats(tweet.split()), 'negative') for tweet in negative_tweets]
neutralfeats = [(word_feats(tweet.split()), 'neutral') for tweet in neutral_tweets]

# separate 75% for train, 25% for test
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
neutralcutoff = len(neutralfeats)*3/4

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] + neutralfeats[:neutralcutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] + neutralfeats[neutralcutoff:]
print 'train on %d instances, test on %d instances' % (
    len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()