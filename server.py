from flask import Flask,request,render_template,jsonify,json

import sentiment as sentimentinterface
import evaluate as spooky
import classify 
import timeit
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import tarfile

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

#import nltk
#from nltk.corpus import stopwords
#stopWords = set(stopwords.words('english'))

print("Reading data")
tarfname = "data/sentiment.tar.gz"
sentiment =  sentimentinterface.read_files(tarfname)

print("\nTraining classifier")
cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
print("\nEvaluating")
classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

tf_train, y_train, tf_vect = spooky.processing()
clf = MultinomialNB()
clf = spooky.trainMNB(clf, tf_train, y_train)

names = ['Edgar Allan Poe', 'H.P. Lovecraft', 'Mary Wollstonecraft Shelley']
set_names = ['EAP', 'HPL', 'MWS']

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spooky')
def blog():
    return render_template('spooky.html')


@app.route('/visual',methods = ['POST', 'GET'])
def visual():

    text = request.form['test_text']
    query = sentiment.tfidf.transform([text])
    output = cls.predict(query)[0]
    confidence = cls.predict_proba(query)[0] * 100
    confidence = np.around(confidence, decimals=2)
    if output:
        output = "POSITIVE REVIEW"
    else:
        output = "NEGATIVE REVIEW"
    
    set_names = ['coef']
    _,inds = np.nonzero(query.toarray())
    coef_data = []
    prod_data = []
    for i in range(cls.coef_.shape[0]):
        coef_data.append(list(cls.coef_[i, inds]))
        prod_data.append(list(cls.coef_[i, inds] * query.toarray()[0, inds]))

    features = np.array(sentiment.tfidf.get_feature_names())[inds]
    features = [x.encode('ascii') for x in features]
    uni = [x for x in features if len(x.split()) == 1]
    uni_ind = [i for i,x in enumerate(features) if len(x.split()) == 1]
    uni_data = list(np.array(prod_data)[:,uni_ind])
    uni_data = [list(x) for x in uni_data]

    query = list(query.toarray()[0, inds[uni_ind]])
    return render_template('visual.html', text=text, output=output, confidence=confidence, \
                           labels=features, set_names=set_names, data=coef_data, uni=uni, uni_data=uni_data, query=query, prod_data=prod_data)

@app.route('/analysis',methods = ['POST', 'GET'])
def analysis():
    text = request.form['test_text']

    query = tf_vect.transform([text])
    output = clf.predict(query)[0]
    confidence = clf.predict_proba(query)[0] * 100
    confidence = np.around(confidence, decimals=2)
    
    category = 0
    for i,n in enumerate(set_names):
        if n == output:
            output = names[i]
            category = i
            break

    _,inds = np.nonzero(query.toarray())
    data = []
    for i in range(clf.coef_.shape[0]):
        data.append(list(clf.coef_[i, inds] * query.toarray()[0, inds]))  

    features = np.array(tf_vect.get_feature_names())[inds]
    features = [x.encode('ascii') for x in features]
    uni = [x for x in features if len(x.split()) == 1]
    uni_ind = [i for i,x in enumerate(features) if len(x.split()) == 1]
    uni_data = list(np.array(data)[:,uni_ind])
    uni_data = [list(x) for x in uni_data]

    query = query.toarray()[0]
    uni_query = list(query[inds[uni_ind]])

    
    return render_template('analysis.html', text=text, output=output, confidence=confidence, n=category, \
                            labels=features, set_names=set_names, data=data, uni=uni, uni_data=uni_data, query=uni_query)

if __name__ == '__main__':
    app.run(debug=True)