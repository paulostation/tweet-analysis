import pandas as pd
import numpy as np

df = pd.read_csv('/home/paulostation/Documents/lula.csv', names=['sentiment', 'text']);

# separate tweet text only
tweets = df["text"].tolist()
# separate positive tweets only
positive_tweets = df.loc[df['sentiment'] == 'positive']['text'].tolist()
print positive_tweets[0]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(positive_tweets) 

pos_words = vectorizer.get_feature_names()

print pos_words

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


def word_feats(words):
        return dict([(word, True) for word in words])

# print tweets[0]

# print data_corpus[0]
# print type(positive_tweets)
# print type(data_corpus)

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
# posfeats = [((word_feats(word, 'pos')) for word in pos_words]

# print negfeats[0]

# print pos_words
# print movie_reviews.words(negids[0])



# negcutoff = len(negfeats)*3/4
# poscutoff = len(posfeats)*3/4

# trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
# testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
# print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

# classifier = NaiveBayesClassifier.train(trainfeats)
# print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
# classifier.show_most_informative_features()