from flask import Flask, Response, request
import os
from TweetClassifier import classify

app = Flask(__name__)

port = int(os.getenv("VCAP_APP_PORT", 5000))

def root_dir():
    return os.path.abspath(os.path.dirname(__file__))

@app.route('/classify',methods=['GET', 'POST'])
def classify_tweet():
    text = request.args["text"]
    
    return classify(text)
    
if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=port)
