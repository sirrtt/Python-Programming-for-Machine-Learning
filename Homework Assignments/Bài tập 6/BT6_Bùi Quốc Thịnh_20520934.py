# Họ và tên: Bùi Quốc Thịnh
# Mã số sinh viên: 20520934

from csv import list_dialects
import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

st.title("Machine Learning Website")
st.header("1. Upload dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = "data/" + uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data) 

    st.header("2. Display dataset")
    dataframe = pd.read_csv(df)
    st.write(dataframe)
    
    st.header("3. Choose input features")
    X = dataframe.iloc[:, :-1]
    for i in X.columns:
        agree = st.checkbox(i)
        if agree == False:
            X = X.drop(i, 1)
    st.write(X)
    flag = 0
    for i in X.columns:
        if X[i].dtypes == object:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
            flag = 1
    
    st.header("3.1. Outputs")
    y = dataframe.iloc[:, -1]
    st.write(y)

    st.header("4. Choose hyper parameters")
    train_per = st.slider(
        'Select a range of training dataset',
        0, 100, 80)
    st.write('The training dataset is ', train_per,'%')
    st.write('Therefore, the test dataset is ', 100 - train_per,'%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_per) * 0.01, random_state=0)

    st.header("5. Metrics")
    while (True):
        f1score = st.checkbox('F1-score')
        logloss = st.checkbox('Log loss')
        if f1score == True or logloss == True:
            break
    
    st.header("6. Choose K-Fold Cross-validation or not")
    k_fold = st.checkbox('K-Fold Cross-validation')
    if k_fold == True:
        num = st.number_input('Insert the number of fold:')
        st.write('The number is ', num)
        num = int(num)

    if st.button('Run'):
        st.write('Linear Regression init')
        pipeline = make_pipeline(StandardScaler(), LogisticRegression())
        df_me = pd.DataFrame(columns = ['F1-score', 'Log loss'])

        if k_fold == True:
            folds = KFold(n_splits = num, shuffle = True, random_state = 100)
            scores = cross_val_score(pipeline, X_train, y_train, scoring='f1', cv=folds)
            scores_2 = cross_val_score(pipeline, X_train, y_train, scoring='neg_log_loss', cv=folds)
            for i in range(len(scores)):
                df_me = df_me.append({'F1-score' : scores[i], 'Log loss' : abs(scores_2[i])}, ignore_index = True)
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            log = log_loss(y_test, y_pred)
            df_me = df_me.append({'F1-score' : f1, 'Log loss' : abs(log)}, ignore_index = True)
        if f1score and logloss:
            st.write(df_me)
            st.bar_chart(df_me['F1-score'])
            st.bar_chart(df_me['Log loss'])
        elif f1score:
            st.write(df_me['F1-score'])
            st.bar_chart(df_me['F1-score'])
        else:
            st.write(df_me['Log loss'])
            st.bar_chart(df_me['Log loss'])