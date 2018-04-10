# tweet-analysis

A simple flask application that uses nltk NaiveBayesClassifier to analyze a tweet in Brazilian Portuguese and return if it is whether a positive or negative tweet.

## Usage

Requirements:
 - Python 2.7 installed
 - Pip installed and setup

1. Run `pip install -r requirements.txt`
2. Run `python app.py`
3. Access http://localhost:5000/classify?text=Esse%20tweet%20%C3%A9%20muito%20bom to test

TODO

 - [X] Make classifier work with csv file
 - [X] Modularize app
 - [X] Add Continuous Deployment to the app
 - [X] Add tests
 - [ ] Filter out words that are in both feature sets, as they don't differentiate features
 - [ ] Test usage section