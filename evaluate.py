#! /usr/bin/python

import numpy as np
import pandas as pd
import string
import sys


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# customize tfidf vectorizer with stemming 
def tokenize(text):
    #translator = str.maketrans('', '', string.punctuation)
    #text = text.lower().translate(translator)
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = nltk.word_tokenize(text.translate(str.maketrans('', '', string.punctuation)))
    stems = [stemmer.stem(item) for item in tokens]
    return stems

def processing(path='train.csv'):
    df = pd.read_csv(path)
    #df['polarity'] = df['text'].map(lambda text: TextBlob(text).sentiment.polarity)
    df['length'] = df['text'].apply(lambda x: len(str(x).split()))
    #df['num_unique'] = df['text'].apply(lambda x: len(set(str(x).split())))
    #df["num_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopWords]))
    #df["num_punctuations"] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    #df["mean_word_len"] = df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df = df[df['length'] < 200]
    
    # splitting train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df, df.author.values, test_size=0.1)
    X_train_text = X_train.text.values
    #X_test_text = X_test.text.values
    #X_train_meta = X_train[['polarity','length','num_unique','num_stopwords','num_punctuations','mean_word_len']]
    #X_test_meta = X_test[['polarity','length','num_unique','num_stopwords','num_punctuations','mean_word_len']]
    
    # vectorizing text
    tf_vect = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', decode_error='ignore')
    corpus = pd.concat([X_train['text'],X_test['text']])
    tf_vect.fit(corpus)
    tf_train = tf_vect.transform(X_train_text)
    #tf_test = tf_vect.transform(X_test_text)
    
    # stacking text data and meta data
    #X_train_all = sparse.hstack([tf_train, sparse.csr_matrix(X_train_meta)]).tocsr()
    #X_test_all = sparse.hstack([tf_test, sparse.csr_matrix(X_test_meta)]).tocsr()
    
    return tf_train, y_train, tf_vect

def trainMNB(clf, X_train, y_train):
    cv_scores = []
    kfold = KFold(n_splits=3, shuffle=False, random_state=None)
    for train_indices, val_indices in kfold.split(X_train):
        train_X = X_train[train_indices]
        train_y = y_train[train_indices]
        val_X = X_train[val_indices]
        val_y = y_train[val_indices]
        clf.fit(train_X, train_y)
        pred_y = clf.predict_proba(val_X)
    
    return clf

def test(tf_vect, clf, sent):
    sent_v = tf_vect.transform(sent)
    result = clf.predict(sent_v)
    result_prob = clf.predict_proba(sent_v)
    return result, result_prob

def get_coef(dic, features, index, s):
    arr = []
    for w in s:
        coef = features[index][dic[w.lower()]] if w.lower() in dic else 0
        arr.append(coef)
    return arr

def graph(tf_vect, clf, sent):
    test_df = pd.DataFrame()
    features = clf.feature_log_prob_
    dic = tf_vect.vocabulary_
    s = "".join((char if char.isalpha() else " ") for char in sent[0]).split()
    test_df['words'] = s
    test_df['coef_eap'] = get_coef(dic, features, 0, s)
    test_df['coef_hpl'] = get_coef(dic, features, 1, s)
    test_df['coef_mws'] = get_coef(dic, features, 2, s)
    
    df_eap = pd.DataFrame(test_df.coef_eap.values, index=test_df.words.values)
    plt.figure(figsize = (2,20))
    sns.heatmap(df_eap, annot=True, fmt="g", cmap='viridis')
    plt.savefig('heatmap_EAP.png', bbox_inches = 'tight')
    #plt.show()
    
    df_hpl = pd.DataFrame(test_df.coef_hpl.values, index=test_df.words.values)
    plt.figure(figsize = (2,20))
    sns.heatmap(df_hpl, annot=True, fmt="g", cmap='plasma')
    plt.savefig('heatmap_HPL.png', bbox_inches = 'tight')
    #plt.show()
    
    df_mws = pd.DataFrame(test_df.coef_mws.values, index=test_df.words.values)
    plt.figure(figsize = (2,20))
    sns.heatmap(df_mws, annot=True, fmt="g", cmap='inferno')
    plt.savefig('heatmap_MWS.png', bbox_inches = 'tight')
    #plt.show()

def main():
    sent = []
    for arg in sys.argv[1:]:
        sent.append(arg)
    tf_train, y_train, tf_vect = processing()
    clf = MultinomialNB()
    clf = trainMNB(clf, tf_train, y_train)
    result, result_prob = test(tf_vect, clf, sent)
    if result[0] == 'EAP':
        print("Conguatualations! You are the next Edgar Allan Poe!")
    elif result[0] == 'HPL':
        print("Conguatualations! You are the next HP Lovecraft!")
    else:
        print("Conguatualations! You are the next Mary Wollstonecraft Shelley!")
    graph(tf_vect, clf, sent)

if __name__ == "__main__":
    main()


