import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tkinter import messagebox

########################################################################################################################
st.title("Gold Price Prediction using Machine Learning")

########################################################################################################################
if st.button('clean data'):

    data = pd.read_csv(r"Dataset.csv")

    st.write(f"the UnCleaned Data Null values", data.isnull().sum())

    fig0, ax2 = plt.subplots()
    sns.heatmap(data.isnull(), yticklabels=False, cmap='coolwarm')
    st.write(f"the Cleaned Data Description", fig0)

    data.dropna(inplace=True)
    st.write(f"the Cleaned Data Null values", data.isnull().sum())
    st.write(f"the Cleaned Data Shape", data.shape)

    st.info('Cleaning Data Completed!')
########################################################################################################################

if st.button('Train'):

    st.write(""" The choice between using RandomForestRegressor or RandomForestClassifier 
    depends on the nature of your prediction task. Since you want to predict the price of gold, 
    which is a continuous variable, RandomForestRegressor is the more appropriate choice.""")


    st.info('Train Started. Kindly wait......')
    newdf = pd.read_csv("updated_data.csv", index_col=0)

    X = newdf.loc[:9431, ['Close', 'Volume', 'Open']]
    Y = newdf.loc[:9431, ['High', 'Low']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

    st.write("Total data Sizes : ", X.shape)

    model = RandomForestRegressor(n_estimators=100)

    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)

    st.write(f'Mean Absolute Error (MAE): {mae}')
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')
    st.write(f'R-squared (R2): {r2}')

    st.write(f"This Model scores: {round(r2 * 100, 2)}% accuracy")

    st.info('Train Completed!')

########################################################################################################################

if st.button('Show Data Anaytics'):

    newdf = pd.read_csv("updated_data.csv", index_col=0)

    fig, ax = plt.subplots()
    sns.heatmap(newdf.isnull(), yticklabels=False, cmap='coolwarm')
    st.write(f"the Cleaned Data Description", fig)

    fig1, ax1 = plt.subplots()
    sns.distplot(newdf['High'],color='purple')
    st.write(f"the Cleaned Data Description", fig1)

    fig2, ax2 = plt.subplots()
    sns.distplot(newdf['Volume'], color='Blue')
    st.write(f"the Cleaned Data Description", fig2)
    
########################################################################################################################

if st.button('Show Confusion Matrix, Precision, Recall'):

    st.warning("""The ML Model used here is Regression type of Machine Learning Model, 
    Confusion matrices are typically used for classification problems, 
    where the goal is to predict categories or classes. 
    For continuous quantity data, confusion matrices are not directly applicable 
    because the predictions are not categorical. 
    Instead, for regression problems where you are predicting a continuous numeric value, 
    you would typically use regression evaluation metrics.\n
    Common regression metrics include:\n

    1.Mean Absolute Error (MAE)\n
    2.Mean Squared Error (MSE)\n
    3.Root Mean Squared Error (RMSE)\n """)

    st.warning("""Precision and recall are metrics commonly used in classification problems and 
    are not directly applicable to regression problems, where the goal is to predict a continuous quantity. 
    Precision and recall are designed to evaluate the performance of models 
    in correctly classifying instances into different categories.

    For regression problems, you typically use metrics such as Mean Absolute Error (MAE), 
    Mean Squared Error (MSE), and R-squared to evaluate the performance of models. 
    These metrics assess the accuracy of the predicted numerical values compared to the actual values. """)

    if st.button('OK'):
        rev = messagebox.askyesno(message="Was this information Useful?")
        if rev == 1:
            messagebox.showinfo(message="Thank You")

########################################################################################################################
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Start', on_click=click_button)

if st.session_state.clicked:
    st.markdown("Please input the requirement, to predict the high and low price of Gold")

    close = st.slider("Select the Close price of Gold in USD", 0, 5000)
    vol = st.slider("Select the Sum of stockings of Gold Commodity", 0, 5000)
    openn = st.slider("Select the Open price of Gold on that day", 0, 5000)

    st.write(f"Close price of Gold is: {close}")
    st.write(f"Sum of stockings of Gold Commodity is: {vol}")
    st.write(f"Open price of Gold on that day is: {openn}")

    if st.button('Predict'):
        newdf = pd.read_csv("updated_data.csv", index_col=0)

        X = newdf.loc[:9431, ['Close', 'Volume', 'Open']]
        Y = newdf.loc[:9431, ['High', 'Low']]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

        st.write("Total data Sizes : ", X.shape)

        model = RandomForestRegressor(n_estimators=100)

        model.fit(X_train, Y_train)

        sample = [[close, vol, openn]]
        array_pred = model.predict(sample)

        pred = array_pred.tolist()

        st.write(f"the lowest price of gold predicted is: INR {(pred[0][1]) * 85 } per 40 grams")

        st.write(f"the highest price of gold predicted is: INR {(pred[0][0]) * 85} per 40 grams")


        st.write("Thank you")
########################################################################################################################

if st.button("Exit"):
    st.success("Thank you")
    st.markdown("""
        <meta http-equiv="refresh" content="0; url='https://www.google.com'" />
        """, unsafe_allow_html=True)
