import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(page_title='CSV File Reader', layout='wide')
st.header('File Upload')

# uploaded_file = st.file_uploader("Choose a file")

# def extract(file_to_extract):
#     if file_to_extract.name.split(".")[-1] == "csv": 
#         extracted_data = pd.read_csv(file_to_extract)

#     elif file_to_extract.name.split(".")[-1] == 'json':
#          extracted_data = pd.read_json(file_to_extract, lines=True)

#     elif file_to_extract.name.split(".")[-1] == 'xml':
#          extracted_data = pd.read_xml(file_to_extract)
         
#     return extracted_data

# create an empty list which will be used to merge the files.

# df =  extract(uploaded_file)
df = pd.read_csv(r"https://raw.githubusercontent.com/hotmail8848/review/main/Amazone%20reviews.csv")

if 'df' not in st.session_state:
    st.session_state['df'] = df

st.dataframe(df, width=1800, height=1200)
