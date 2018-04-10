# encoding=utf8
import pandas as pd
import numpy as np
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

classifier = None

stopwords = set(["nd","teve","alm","porque","vo","era","to","nos","nas","que","os","ela","ou","meu","ja","me","hj","quem","vou","ele","ir","na","em","at","ao","c","todos","dos","est","mas","diz","sem","sua","e","sobre","se","voc","isso","q","do","vai","pra","a","o","dilma","lula","bolsonaro","no","quando","cara","uma","so","um","da","ato","faz","as","esto","mais","com","foi","mesmo","das","quer","fazer","eu"])

def pre_process(word):
    # remove upper case letters
    word = word.lower()
    # remove urls
    word = re.sub(r'https:\/\/[^\s]+', r'', word)
    # remove diacritics
    word = word.encode("ascii","ignore")
    # remove symbols
    word = re.sub(r'[^A-z\ ]', r'', word)
    
    return word

def word_feats(words):
    return dict([(word, True) for word in words if word.lower() not in stopwords and len(word) > 1])

# Function used to classify tweets
def classify(text):
    features = word_feats(text.split())
    classification = classifier.classify(features)
    return classification

df = pd.read_csv('datasets/labeled_tweets.csv', names=["text","sentiment"],encoding='utf8');

# separate positive tweets only
positive_tweets = df.loc[df['sentiment'] == 'positive']['text'].tolist()
# separate negative tweets only
negative_tweets = df.loc[df['sentiment'] == 'negative']['text'].tolist()
# separate neutral tweets only
neutral_tweets = df.loc[df['sentiment'] == 'neutral']['text'].tolist()

print("Number of positive tweets: {}.".format(len(positive_tweets)))
print("Number of negative tweets: {}.".format(len(negative_tweets)))
print("Number of neutral tweets: {}.".format(len(neutral_tweets)))

positive_tweets = map(pre_process, positive_tweets)
negative_tweets = map(pre_process, negative_tweets)
neutral_tweets = map(pre_process, neutral_tweets)

posfeats = [(word_feats(tweet.split()), 'positive') for tweet in positive_tweets]
negfeats = [(word_feats(tweet.split()), 'negative') for tweet in negative_tweets]
neutralfeats = [(word_feats(tweet.split()), 'neutral') for tweet in neutral_tweets]

# separate 75% for train, 25% for test
negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
neutralcutoff = len(neutralfeats)*3/4

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]  + neutralfeats[:neutralcutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]  + neutralfeats[neutralcutoff:]
print 'train on %d instances, test on %d instances' % (
    len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features(n=100)