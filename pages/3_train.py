import pandas as pd 
import numpy as np 

import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

df = st.session_state.unchanged_df

X = df['Review']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 42,
                                                   test_size = 0.20)

clf = Pipeline([
    ('vect', CountVectorizer(stop_words= "english")),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
    ])

fit_model = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

st.write('Training accuracy:', fit_model.score(X_train,y_train))
st.write('Test accuracy:', fit_model.score(X_test,y_test))

st.write(classification_report(y_test,y_pred))

st.write(confusion_matrix(y_test,y_pred))
