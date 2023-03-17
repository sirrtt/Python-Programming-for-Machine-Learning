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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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
        MAE = st.checkbox('MAE')
        MSE = st.checkbox('MSE')
        if MAE == True or MSE == True:
            break
    
    st.header("6. Choose K-Fold Cross-validation or not")
    k_fold = st.checkbox('K-Fold Cross-validation')
    if k_fold == True:
        num = st.number_input('Insert the number of fold:')
        st.write('The number is ', num)
        num = int(num)

    if st.button('Run'):
        st.write('Linear Regression init')
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        df_me = pd.DataFrame(columns = ['MAE', 'MSE'])
        if k_fold == True:
            folds = KFold(n_splits = num, shuffle = True, random_state = 100)
            scores = cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error', cv=folds)
            scores_2 = cross_val_score(lr, X_train, y_train, scoring='neg_mean_squared_error', cv=folds)
            for i in range(len(scores)):
                df_me = df_me.append({'MAE' : abs(scores[i]), 'MSE' : abs(scores_2[i])}, ignore_index = True)
            labels = []
            for i in range(num):
                labels.append(str(i+1) + ' Fold')
            X_axis = np.arange(len(labels))
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            df_me = df_me.append({'MAE' : mae, 'MSE' : mse}, ignore_index = True)
            X_axis = np.arange(1)

        if MAE == True and MSE == True:
            st.write(df_me)
            fig, ax = plt.subplots(figsize=(20, 20))
            plt.bar(X_axis - 0.5/2, df_me['MAE'], width=0.5, color='red', label='MAE')
            plt.bar(X_axis + 0.5/2, df_me['MSE'], width=0.5, color='blue', label='MSE')
            if k_fold == True:
                plt.xticks(X_axis, labels)
            plt.title('Compare MAE and MSE', fontsize=30)
            plt.xlabel('Linear Regression', fontsize=15)
            plt.ylabel('Metrics', fontsize=15)
            plt.yscale('log')
            plt.legend()
            fig.tight_layout()
            plt.grid(True)
            st.pyplot(fig)  
        elif MAE == True:
            st.write(df_me['MAE'])
            st.bar_chart(df_me['MAE'])
        else:
            st.write(df_me['MSE'])
            st.bar_chart(df_me['MSE'])