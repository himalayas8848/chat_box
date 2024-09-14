import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re 
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud

# df = pd.read_csv(r'C:\Users\dv\Downloads\Amazone reviews.csv')
df = st.session_state['df']

df = df.sample(frac= 0.20,replace = True).reset_index(drop = True)
df.dropna(inplace= True)

df = pd.DataFrame(df,columns = ['Score','Text'])
df.rename(columns = {'Score':'Rating','Text':'Review'},inplace = True)

def apply_sentiment(Rating):
    if(Rating <=2 ):
        return 0
    else:
        return 1
    
df['Sentiment'] = df['Rating'].apply(apply_sentiment)

sentiment = df['Sentiment'].value_counts()

def clean_text(Review):
   
    Review = str(Review).lower() # convert to lowercase
    Review = re.sub('\[.*?\]', '', Review) 
    Review = re.sub('https?://\S+|www\.\S+', '', Review) # Remove URls
    Review = re.sub('<.*?>+', '', Review)
    Review = re.sub(r'[^a-z0-9\s]', '', Review) # Remove punctuation
    Review = re.sub('\n', '', Review)
    Review = re.sub('\w*\d\w*', '', Review)
    return Review

df['Review'] = df['Review'].apply(clean_text)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stopword = []
sentence = df['Review'][0]


#words = nltk.word_tokenize(sentence)


def remove_stopword(stop_words, sentence):
    return [word for word in nltk.word_tokenize(sentence) if word not in stop_words]

df['reviews_text'] = df['Review'].apply(lambda row: remove_stopword(stop_words, row))

rating_pct = df['Rating'].value_counts()/len(df) * 100
rating_pct_plot = rating_pct.reset_index()
rating_pct_plot.columns = ['Rating', 'Count']

fig_bar = px.bar(
    rating_pct_plot,
    x="Rating",
    y="Count",
)

df['Word_Count'] = df['Review'].apply(lambda x: len(x.split(' ')))

fig_hist = px.histogram(
    df,
    x="Word_Count",
)

tab1, tab2 = st.tabs(["Rating Status", "Numbers of Words"])
with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig_bar, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(fig_hist, use_container_width=True)

if "unchanged_df" not in st.session_state:
     st.session_state["unchanged_df"] = df
