import pickle
import streamlit as st
import numpy as np
import pandas as pd
import base64
from peopleanalytics.PeopleAnalytics import PeopleAnalytics

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="analytics_results.csv">Download csv file</a>'

# sets container max-width
st.markdown("""<style>.reportview-container .main .block-container{max-width: 1280px;}</style>""", unsafe_allow_html=True)

# loads model
model = pickle.load(open('model/model_people_analytics.pkl', 'rb'))


"""
# People Analytics :busts_in_silhouette:
This tool helps the HR team to have a list of identification numbers whose employees are most likely to leave the company.

Among the information about the employees, there are several others, such as, department, education field, job satisfaction, job role, rates, etc.

## How it works

1. Upload a .csv file.
2. Check the prediction results.
3. Download the results as a .csv file to be imported in other systems and applications.

"""

st.markdown(f'Download the test data csv file <a href="https://app.box.com/s/4uuf67pqgeom5rjw66luq7heg00qm2fg" target="_blank" download="test_data.csv">here</a> to test the tool. You may want to make some changes on it too.', unsafe_allow_html=True)

"""
---
## 1. Upload .csv file :arrow_up:
Browse or drop a .csv file in the shaded area.
"""
uploaded_file = st.file_uploader(" ", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write(' This table represents the data from the uploaded file. There are {} records.'.format(data.shape[0]))
    st.write('When you hover on the table, it appears an expand icon on the top right corner of it. So you can have a better visualization.')

    st.dataframe(data, height=540)
    
    # instantiates PeopleAnalytics class
    pipeline = PeopleAnalytics()

    # data cleaning
    df1 = pipeline.data_cleaning(data)

    # feature engineerin
    df2 = pipeline.feature_engineering(df1)

    # data preparation
    df3 = pipeline.data_preparation(df2)

    # prediction
    df_response = pipeline.get_prediction(model, data, df3)

    """
    ---
    ## 2. Check the prediction results :clipboard:
    """
    st.write(' This table represents the data from the predictions. There are {} records which describe the employees that are most likely to leave the company.'.format(df_response.shape[0]))
    st.write('When you hover on the table, it appears an expand icon on the top right corner of it. So you can have a better visualization.')

    st.dataframe(df_response.reset_index().drop('index', axis=1), height=540)

    
    """
    ---
    ## 3. Download the results :arrow_down:
    """

    st.markdown(get_table_download_link(df_response), unsafe_allow_html=True)

else:
    st.write('No file was uploaded.')
